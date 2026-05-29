import pandas as pd
import numpy as np
import random
import boto3
import torch
import multiprocessing
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from decimal import Decimal


# Configuro el aws
dynamodb = boto3.resource(
    'dynamodb',
    region_name='us-east-1'
)

# Convierto las columnas decimal a float o entero segun corresponda
def convert_decimals(obj):

    if isinstance(obj, list):
        return [convert_decimals(i) for i in obj]

    elif isinstance(obj, dict):
        return {k: convert_decimals(v) for k, v in obj.items()}

    elif isinstance(obj, Decimal):

        if obj % 1 == 0:
            return int(obj)

        return float(obj)

    return obj

# Descargo la tabla y lo paso a df
def get_table_df(table_name):

    table = dynamodb.Table(table_name)
    response = table.scan()
    data = response['Items']
    
    # Manejo para tablas grandes
    while 'LastEvaluatedKey' in response:

        response = table.scan(
            ExclusiveStartKey=response['LastEvaluatedKey']
        )
        data.extend(response['Items'])
    data = [convert_decimals(x) for x in data]

    return pd.DataFrame(data)


# Transformacion para los inputs
def get_inputs_df(table_name='inputs_numeros'):

    df = get_table_df(table_name)
    # Elimino el ID que no me interesa
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    # Transformor a float porque todos son decimales
    df = df.astype('float32')

    return df


# Transformo los outpus segun el tipo de variable
def get_outputs_df(table_name='outputs'):

    df = get_table_df(table_name)
    # Detecto las etiquetas porque son float todas
    etiqueta_cols = [
        c for c in df.columns
        if c.startswith('Etiqueta_')
    ]
    # Solo me interesa adaptar 3 columnas mas, porque el resto no las uso
    columnas_finales = (
        [
            'Fila Noticia',
            'Tickers Mapeados',
            'Date'
        ]
        + etiqueta_cols
    )
    df = df[columnas_finales].copy()

    # Transformo a entero
    df['Fila Noticia'] = (
        pd.to_numeric(df['Fila Noticia'])
        .astype('int64')
    )
    # Transformo a string
    df['Tickers Mapeados'] = (
        df['Tickers Mapeados']
        .astype(str)
    )
    # Transformo a datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Las etiquetas todas a float
    for col in etiqueta_cols:

        df[col] = (
            pd.to_numeric(df[col], errors='coerce')
            .astype('float32')
        )

    return df

# Necesito para entrenar varios modelos sin problemas
torch._dynamo.disable()


# Defino variables globales para guardar los outputs de interes
arquitectura_NN = None
arquitectura_ensemble = None
metricas_totales = None
diccionario_inferencia_top50 = None

# Funcion para entrenar distintas configuraciones por modelo y hacer sus inferencias
def train_and_inference():
    global arquitectura_NN, arquitectura_ensemble, metricas_totales, diccionario_inferencia_top50
    print("Cargo las tablas de inputs y outputs")
    # Inputs 
    df_inputs = get_inputs_df()

    # Outputs 
    etiquetas = get_outputs_df()

    # Guardo los nombre de cada etiqueta que depende de su longitud de ventata e inicia de esta misma
    columnas_etiqueta = [col for col in etiquetas.columns if col.startswith("Etiqueta")]
    columnas_etiqueta

    print("Inicio el proceso de busqueda de mejor arquitectura para cada modelo y ventana")
    # Modelos ensemble
    # Cambio las etiquetas para el modelo y pongo una semillla para repetir resultados
    mapping = {-1: 0, 1: 1}
    random.seed(42)

    # Combinaciones de hiperparametros de LIGHTGBM para la fase de busqueda amplia
    exploration_grid_lgbm = [
        {"learning_rate": 0.10, "n_estimators": 100,  "max_depth": 3, "num_leaves": 7,  "min_child_samples": 30},
        {"learning_rate": 0.10, "n_estimators": 200,  "max_depth": 4, "num_leaves": 15, "min_child_samples": 20},
        {"learning_rate": 0.05, "n_estimators": 300,  "max_depth": 5, "num_leaves": 31, "min_child_samples": 20},
        {"learning_rate": 0.05, "n_estimators": 500,  "max_depth": 6, "num_leaves": 63, "min_child_samples": 15},
        {"learning_rate": 0.03, "n_estimators": 500,  "max_depth": 5, "num_leaves": 31, "min_child_samples": 40},
        {"learning_rate": 0.03, "n_estimators": 700,  "max_depth": 6, "num_leaves": 63, "min_child_samples": 50},
        {"learning_rate": 0.01, "n_estimators": 1000, "max_depth": 4, "num_leaves": 15, "min_child_samples": 50},
        {"learning_rate": 0.01, "n_estimators": 1500, "max_depth": 5, "num_leaves": 31, "min_child_samples": 30},
    ]

    # Combinaciones de hiperparametros de XGBOOST para la fase de busqueda amplia
    exploration_grid_xgb = [
        {"learning_rate": 0.10, "n_estimators": 100,  "max_depth": 3, "min_child_weight": 1},
        {"learning_rate": 0.10, "n_estimators": 200,  "max_depth": 4, "min_child_weight": 1},
        {"learning_rate": 0.05, "n_estimators": 300,  "max_depth": 4, "min_child_weight": 3},
        {"learning_rate": 0.05, "n_estimators": 500,  "max_depth": 5, "min_child_weight": 3},
        {"learning_rate": 0.03, "n_estimators": 500,  "max_depth": 5, "min_child_weight": 5},
        {"learning_rate": 0.03, "n_estimators": 700,  "max_depth": 6, "min_child_weight": 5},
        {"learning_rate": 0.02, "n_estimators": 1000, "max_depth": 5, "min_child_weight": 8},
        {"learning_rate": 0.01, "n_estimators": 1500, "max_depth": 4, "min_child_weight": 10},
    ]

    # Combinaciones de hiperparametros de LIGHTGBM para la fase de busqueda en el vecindario del mejor modelo de la fase anterior
    def generate_local_params_lgbm(best_params, n_samples=12):
        lr = np.linspace(best_params["learning_rate"] * 0.7, best_params["learning_rate"] * 1.3, n_samples)
        n_est = np.linspace(best_params["n_estimators"] * 0.7, best_params["n_estimators"] * 1.3, n_samples).astype(int)
        depth = np.linspace(max(3, best_params["max_depth"] - 2), best_params["max_depth"] + 2, n_samples).astype(int)
        leaves = np.linspace(max(7, best_params["num_leaves"] * 0.5), best_params["num_leaves"] * 1.5, n_samples).astype(int)
        child = np.linspace(max(5, best_params["min_child_samples"] - 15), best_params["min_child_samples"] + 15, n_samples).astype(int)
        return [{"learning_rate": round(float(lr[i]), 4), "n_estimators": int(n_est[i]),
                "max_depth": int(depth[i]), "num_leaves": int(leaves[i]), "min_child_samples": int(child[i])}
                for i in range(n_samples)]

    # Combinaciones de hiperparametros de XGBOOST para la fase de busqueda en el vecindario del mejor modelo de la fase anterior
    def generate_local_params_xgb(best_params, n_samples=12):
        lr = np.linspace(best_params["learning_rate"] * 0.7, best_params["learning_rate"] * 1.3, n_samples)
        n_est = np.linspace(best_params["n_estimators"] * 0.7, best_params["n_estimators"] * 1.3, n_samples).astype(int)
        depth = np.linspace(max(3, best_params["max_depth"] - 2), best_params["max_depth"] + 2,  n_samples).astype(int)
        child = np.linspace(max(1, best_params["min_child_weight"] - 4), best_params["min_child_weight"] + 4, n_samples).astype(int)
        return [{"learning_rate": round(float(lr[i]), 4), "n_estimators": int(n_est[i]),
                "max_depth": int(depth[i]), "min_child_weight": int(child[i])}
                for i in range(n_samples)]

    # Calculo el score global
    def calculate_global_score(report, roc_auc=None):
        r = report["macro avg"]
        return (r["precision"] + r["recall"] + r["f1-score"]) / 3

    # Defino las variables para guardar mis resultados
    ensemble_results_metrics  = []
    ensemble_results_architecture = []
    global_model_id_lgbm = 0
    global_model_id_xgb = 0

    # Empiezo el entrenamiento por cada configuracion para los modelos de ensemble
    for etiqueta in columnas_etiqueta:
        print(f"\nVENTANA: {etiqueta}")
   
        # Uno inputs y outputs, este ultimo segun que etiqueta este
        df_outputs = etiquetas[[etiqueta, "Tickers Mapeados", "Fila Noticia", "Date"]]
        df_final = pd.concat([df_inputs, df_outputs.reset_index(drop=True)], axis=1).dropna()
        df_final = df_final[df_final[etiqueta] != 0]

        # Genero el split para train, val y test
        unique_news_ids = df_final["Fila Noticia"].unique()
        n_unique = len(unique_news_ids)
        train_ids = unique_news_ids[:int(n_unique * 0.70)]
        val_ids = unique_news_ids[int(n_unique * 0.70):int(n_unique * 0.85)]
        train_df = df_final[df_final["Fila Noticia"].isin(train_ids)]
        val_df = df_final[df_final["Fila Noticia"].isin(val_ids)]

        # Reduzco las dimensiones con PCA de los embedding: titulo, contenido, verbo y contexto 
        def fit_pca(train_data, val_data, n=64):
            pca = PCA(n_components=n)
            return pca.fit_transform(train_data), pca.transform(val_data)

        Xtt, Xtv = fit_pca(train_df.iloc[:, 0:768].values,val_df.iloc[:, 0:768].values)
        Xct, Xcv = fit_pca(train_df.iloc[:, 768:1536].values, val_df.iloc[:, 768:1536].values)
        Xvt, Xvv = fit_pca(train_df.iloc[:, 1536:2304].values, val_df.iloc[:, 1536:2304].values)
        Xxt, Xxv = fit_pca(train_df.iloc[:, 2304:3072].values, val_df.iloc[:, 2304:3072].values)
        Xmt = train_df.iloc[:, 3072:3081].values
        Xmv = val_df.iloc[:, 3072:3081].values

        X_train = pd.DataFrame(np.concatenate([Xtt, Xct, Xvt, Xxt, Xmt], axis=1))
        X_val = pd.DataFrame(np.concatenate([Xtv, Xcv, Xvv, Xxv, Xmv], axis=1))

        y_train = np.vectorize(mapping.get)(train_df[etiqueta].astype(int))
        y_val = np.vectorize(mapping.get)(val_df[etiqueta].astype(int))

        # LightGBM
        print("\nEmpiezo por el modelo LIGHTGBM\n")

        # Entreno y calculos las metricas de interes del entrenamieto y validacion
        def run_lgbm(params):
            m = LGBMClassifier(verbosity=-1, subsample=0.8, colsample_bytree=0.7, random_state=42, **params)
            m.fit(X_train, y_train)
            preds = m.predict(X_val)
            probs = m.predict_proba(X_val)[:, 1]
            report = classification_report(y_val, preds, output_dict=True)
            return m, preds, probs, report, roc_auc_score(y_val, probs)

        # Ordeno segun el mejor score global
        exp_res = sorted([{"params": p, "score_global": calculate_global_score(*run_lgbm(p)[3:])}
                        for p in exploration_grid_lgbm], key=lambda x: x["score_global"], reverse=True)
        # Me quedo con el de mayo score global para buscar combinaciones cercanas
        best_params = exp_res[0]["params"]

        # Entreno y calculos metricas para validacion por las combinaciones vecinas
        for params in generate_local_params_lgbm(best_params):
            global_model_id_lgbm += 1
            model_id = f"LGBM_{global_model_id_lgbm}"
            _, preds, probs, report, roc_auc = run_lgbm(params)
            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

            # Guardo las metricas de todos los hiperparametros vecinos
            ensemble_results_metrics.append({
                "model_id": model_id, "modelo": "LGBM", "ventana": etiqueta,
                "accuracy": accuracy_score(y_val, preds),
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
                "precision_class_1": report["1"]["precision"], "recall_class_1": report["1"]["recall"], "f1_class_1": report["1"]["f1-score"],
                "precision_class_0": report["0"]["precision"], "recall_class_0": report["0"]["recall"], "f1_class_0": report["0"]["f1-score"],
                "roc_auc": roc_auc, "score_global": calculate_global_score(report, roc_auc),
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            })
            # Guardo las arquitecturas de los hiperparametros vecinos
            ensemble_results_architecture.append({
                "model_id": model_id, "modelo": "LGBM", "ventana": etiqueta,
                "learning_rate": params["learning_rate"], "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"], "num_leaves": params["num_leaves"],
                "min_child_samples": params["min_child_samples"],
                "subsample": 0.8, "colsample_bytree": 0.7,
                "complexity": params["n_estimators"] * params["num_leaves"],
            })

        # XHBoost
        print("\nContinuo con XGBOOST\n")

        # Entreno y calculos las metricas de interes del entrenamieto y validacion
        def run_xgb(params):
            m = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                            subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=-1, **params)
            m.fit(X_train, y_train, verbose=False)
            preds = m.predict(X_val)
            probs = m.predict_proba(X_val)[:, 1]
            report = classification_report(y_val, preds, output_dict=True)
            return m, preds, probs, report, roc_auc_score(y_val, probs)

        # Ordeno segun el mejor score global
        exp_res = sorted([{"params": p, "score_global": calculate_global_score(*run_xgb(p)[3:])}
                        for p in exploration_grid_xgb], key=lambda x: x["score_global"], reverse=True)
        # Me quedo con el de mayo score global para buscar combinaciones cercanas
        best_params = exp_res[0]["params"]

        # Entreno y calculos metricas para validacion por las combinaciones vecinas
        for params in generate_local_params_xgb(best_params):
            global_model_id_xgb += 1
            model_id = f"XGB_{global_model_id_xgb}"
            _, preds, probs, report, roc_auc = run_xgb(params)
            tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()

            # Guardo las metricas de todos los hiperparametros vecinos
            ensemble_results_metrics.append({
                "model_id": model_id, "modelo": "XGBoost", "ventana": etiqueta,
                "accuracy": accuracy_score(y_val, preds),
                "precision_macro": report["macro avg"]["precision"],
                "recall_macro": report["macro avg"]["recall"],
                "f1_macro": report["macro avg"]["f1-score"],
                "precision_class_1": report["1"]["precision"], "recall_class_1": report["1"]["recall"], "f1_class_1": report["1"]["f1-score"],
                "precision_class_0": report["0"]["precision"], "recall_class_0": report["0"]["recall"], "f1_class_0": report["0"]["f1-score"],
                "roc_auc": roc_auc, "score_global": calculate_global_score(report, roc_auc),
                "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            })
            # Guardo las arquitecturas de los hiperparametros vecinos
            ensemble_results_architecture.append({
                "model_id": model_id, "modelo": "XGBoost", "ventana": etiqueta,
                "learning_rate": params["learning_rate"], "n_estimators": params["n_estimators"],
                "max_depth": params["max_depth"], "min_child_weight": params["min_child_weight"],
                "subsample": 0.8, "colsample_bytree": 0.7, "gamma": 0,
                "complexity": params["n_estimators"] * params["max_depth"],
            })

    # Transformo las listas de todos los resultado a df
    ensemble_metrics      = pd.DataFrame(ensemble_results_metrics)
    ensemble_architecture = pd.DataFrame(ensemble_results_architecture)



    # RED NEURONAL
    # Configuro el uso de cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model_id = 0

    # Preparo los inputs y outputs para pasarlo a la red neuronal
    def prepare_data(etiqueta, df_inputs, etiquetas, batch_size=512):
        df_outputs = etiquetas[[etiqueta, "Tickers Mapeados", "Fila Noticia", "Date"]]
        df_final = pd.concat([df_inputs, df_outputs.reset_index(drop=True)], axis=1).dropna()
        df_final = df_final[df_final[etiqueta] != 0]

        # Split para train, valid y test
        ids  = df_final["Fila Noticia"].unique()
        n = len(ids)
        train_ids = ids[:int(n * 0.70)]
        val_ids = ids[int(n * 0.70):int(n * 0.85)]
        test_ids = ids[int(n * 0.85):]

        train_df = df_final[df_final["Fila Noticia"].isin(train_ids)]
        val_df = df_final[df_final["Fila Noticia"].isin(val_ids)]
        test_df = df_final[df_final["Fila Noticia"].isin(test_ids)]

        def split_inputs(df):
            return (df.iloc[:, 0:768].values, df.iloc[:, 768:1536].values,
                    df.iloc[:, 1536:2304].values, df.iloc[:, 2304:3072].values,
                    df.iloc[:, 3072:3081].values, df[etiqueta].values)

        X_tit_tr, X_con_tr, X_vrb_tr, X_ctx_tr, X_met_tr, y_tr = split_inputs(train_df)
        X_tit_vl, X_con_vl, X_vrb_vl, X_ctx_vl, X_met_vl, y_vl = split_inputs(val_df)
        X_tit_te, X_con_te, X_vrb_te, X_ctx_te, X_met_te, y_te = split_inputs(test_df)

        # Cambio las etiquetas para la red neuronal
        mapping = {-1: 0, 1: 1}
        enc     = lambda y: np.vectorize(mapping.get)(y.astype(int))
        y_tr, y_vl, y_te = enc(y_tr), enc(y_vl), enc(y_te)

        # Estandarizacion de los inputs por tipp
        scalers = [StandardScaler() for _ in range(5)]
        tr_arrs = [X_tit_tr, X_con_tr, X_vrb_tr, X_ctx_tr, X_met_tr]
        vl_arrs = [X_tit_vl, X_con_vl, X_vrb_vl, X_ctx_vl, X_met_vl]

        tr_arrs = [s.fit_transform(x) for s, x in zip(scalers, tr_arrs)]
        vl_arrs = [s.transform(x)     for s, x in zip(scalers, vl_arrs)]

        # Transformo los input a tensores
        t = lambda x: torch.tensor(x, dtype=torch.float32)
        tr_tensors = [t(x) for x in tr_arrs] + [torch.tensor(y_tr, dtype=torch.long)]
        vl_tensors = [t(x) for x in vl_arrs] + [torch.tensor(y_vl, dtype=torch.long)]

        mk_loader = lambda tensors, shuffle: DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle)
        return {"train_loader": mk_loader(tr_tensors, False), "val_loader": mk_loader(vl_tensors, False)}

    # Arquitectura base de la red neuronal
    class MultiInputModel(nn.Module):
        def __init__(self, hidden_dim=128, dropout=0.3):
            super().__init__()
            sd = hidden_dim // 2

            def emb_block(in_d, out_d): return nn.Sequential(
                nn.Linear(in_d, out_d), nn.ReLU(), nn.BatchNorm1d(out_d), nn.Dropout(dropout),
                nn.Linear(out_d, out_d), nn.ReLU())

            def small_block(in_d, out_d): return nn.Sequential(
                nn.Linear(in_d, out_d), nn.ReLU(), nn.Linear(out_d, out_d), nn.ReLU())

            self.title_branch   = emb_block(768, hidden_dim)
            self.content_branch = emb_block(768, hidden_dim)
            self.verb_branch    = small_block(768, sd)
            self.context_branch = small_block(768, sd)
            self.meta_branch    = small_block(9, 32)

            combined_dim = hidden_dim * 2 + sd * 2
            self.gate = nn.Sequential(nn.Linear(combined_dim, combined_dim), nn.Sigmoid())
            self.head = nn.Sequential(
                nn.Linear(combined_dim * 2 + 32, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 1))

        def forward(self, x_title, x_content, x_verb, x_context, x_meta):
            x_title, x_content = F.normalize(x_title, dim=1), F.normalize(x_content, dim=1)
            x_verb, x_context  = F.normalize(x_verb, dim=1),  F.normalize(x_context, dim=1)
            t, c, v, cx = self.title_branch(x_title), self.content_branch(x_content), \
                        self.verb_branch(x_verb), self.context_branch(x_context)
            m        = self.meta_branch(x_meta)
            combined = torch.cat([t, c, v, cx], dim=1)
            gated    = combined * self.gate(combined)
            return self.head(torch.cat([combined, gated, m], dim=1))

    # Combinacion de hiperparametros para la fase de busqueda amploa
    exploration_grid_nn = [
        {"hidden_dim":  64, "dropout": 0.25, "learning_rate": 1e-3, "epochs": 15},
        {"hidden_dim":  96, "dropout": 0.30, "learning_rate": 5e-4, "epochs": 20},
        {"hidden_dim": 128, "dropout": 0.35, "learning_rate": 1e-4, "epochs": 30},
        {"hidden_dim": 160, "dropout": 0.40, "learning_rate": 5e-5, "epochs": 50},
        {"hidden_dim": 192, "dropout": 0.45, "learning_rate": 1e-5, "epochs": 70},
    ]

    # Calcolo de global score
    def calculate_global_score(report):
        r = report["macro avg"]
        return (r["precision"] + r["recall"] + r["f1-score"]) / 3

    # Genero combinacion de hiperaparametros cercano al mejor de la fase de exploracion, pero estos deben tener sentido
    def generate_local_nn_params(best_params):
        bh, bd = best_params["hidden_dim"], best_params["dropout"]
        hiddens  = [int(bh * f) for f in (0.75, 0.90, 1.00, 1.10, 1.25)]
        dropouts = [round(max(0.15, min(0.60, bd + d)), 2) for d in (-0.10, -0.05, 0, 0.05, 0.10)]
        return [{"hidden_dim": h, "dropout": d,
                "learning_rate": best_params["learning_rate"], "epochs": best_params["epochs"]}
                for h, d in zip(hiddens, dropouts)]

    # Entrenamiento de la red neuronal y obtencio de metricas con validacion
    def train_model(params, train_loader, val_loader):
        model     = MultiInputModel(params["hidden_dim"], params["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(params["epochs"]):
            model.train()
            for *xs, y_batch in train_loader:
                xs      = [x.to(device) for x in xs]
                y_batch = y_batch.view(-1, 1).float().to(device)
                optimizer.zero_grad()
                criterion(model(*xs), y_batch).backward()
                optimizer.step()

        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for *xs, y_batch in val_loader:
                xs    = [x.to(device) for x in xs]
                probs = torch.sigmoid(model(*xs))
                preds = (probs > 0.5).int()
                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(y_batch.numpy().flatten())

        report = classification_report(all_labels, all_preds, output_dict=True)
        roc_auc = roc_auc_score(all_labels, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision_macro": report["macro avg"]["precision"],
            "recall_macro":    report["macro avg"]["recall"],
            "f1_macro":        report["macro avg"]["f1-score"],
            "precision_class_1": report["1"]["precision"], "recall_class_1": report["1"]["recall"], "f1_class_1": report["1"]["f1-score"],
            "precision_class_0": report["0"]["precision"], "recall_class_0": report["0"]["recall"], "f1_class_0": report["0"]["f1-score"],
            "roc_auc": roc_auc, "score_global": calculate_global_score(report),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "total_params": sum(p.numel() for p in model.parameters()),
        }

    # Defino los depositvos para las metricas y arquitecturas
    results_metrics      = []
    results_architecture = []

    # Ejecuto el entrenamiento y obtencion de metricas de validacion para cada configuracion de la red neuronal
    for etiqueta in columnas_etiqueta:
        print(f"VENTANA: {etiqueta}")

        data = prepare_data(etiqueta=etiqueta, df_inputs=df_inputs, etiquetas=etiquetas)
        train_loader = data["train_loader"]
        val_loader = data["val_loader"]

        # Exploro las combinaciones amplias y me quedo el de mayor score global
        exp_res = sorted(
            [{"params": p, "score_global": train_model(p, train_loader, val_loader)["score_global"]}
            for p in exploration_grid_nn],
            key=lambda x: x["score_global"], reverse=True)
        best_params = exp_res[0]["params"]
        print("\nMejor configuracion para la red neuronal:", best_params)

        # Continuo con las busqueda de hiperparametro, pero dentro del vencidario de la combinacion ganadora en la fase anterior
        for params in generate_local_nn_params(best_params):
            global_model_id += 1
            model_id = f"NN_{global_model_id}"
            metrics  = train_model(params, train_loader, val_loader)

            results_metrics.append({
                "model_id": model_id, "modelo": "NeuralNetwork", "ventana": etiqueta,
                **{k: v for k, v in metrics.items() if k != "total_params"},
            })
            results_architecture.append({
                "model_id": model_id, "modelo": "NeuralNetwork", "ventana": etiqueta,
                "hidden_dim": params["hidden_dim"], "dropout": params["dropout"],
                "learning_rate": params["learning_rate"], "epochs": params["epochs"],
                "total_params": metrics["total_params"],
            })

    # Guardo las listas de resultados de metricas y arquitecturas de la red neuronal en un df
    NN_metrics      = pd.DataFrame(results_metrics)
    NN_architecture = pd.DataFrame(results_architecture)

    # Uno las metricas de los modelo de emsamble y red neuronal
    total_metrics = pd.concat([ensemble_metrics, NN_metrics], ignore_index=True)
    print("\nFin de la etapa de busqueda de mejor arquitectura por modelo y ventana")



    # INFERENCIA
    print("\nInicio de la etapa de inferencia")
    # Configuro el cpu
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Etiquetas adpatadas para pasarlo por los modelos
    mapping = {-1: 0, 1: 1}
    
    # Preparo los inputs y ouputs para pasarlo correctamente por a red neuronal
    def prepare_data(etiqueta, df_inputs, etiquetas, batch_size=512):
        df_outputs = etiquetas[[etiqueta, "Tickers Mapeados", "Fila Noticia", "Date"]]
        df_final   = pd.concat([df_inputs, df_outputs.reset_index(drop=True)], axis=1).dropna()
        df_final   = df_final[df_final[etiqueta] != 0]
    
        # Split en train, valid y test
        ids       = df_final["Fila Noticia"].unique()
        n         = len(ids)
        train_ids = ids[:int(n * 0.70)]
        val_ids   = ids[int(n * 0.70):int(n * 0.85)]
    
        train_df = df_final[df_final["Fila Noticia"].isin(train_ids)]
        val_df   = df_final[df_final["Fila Noticia"].isin(val_ids)]
    
        slices = [(0, 768), (768, 1536), (1536, 2304), (2304, 3072), (3072, 3081)]
        enc    = lambda y: np.vectorize(mapping.get)(y.astype(int))
    
        tr_arrs = [train_df.iloc[:, a:b].values for a, b in slices]
        vl_arrs = [val_df.iloc[:, a:b].values   for a, b in slices]
        # Estanadirzacion
        scalers = [StandardScaler() for _ in slices]
        tr_arrs = [s.fit_transform(x) for s, x in zip(scalers, tr_arrs)]
        vl_arrs = [s.transform(x)     for s, x in zip(scalers, vl_arrs)]
        # Transformacion a tensores
        t = lambda x: torch.tensor(x, dtype=torch.float32)
        tr_tensors = [t(x) for x in tr_arrs] + [torch.tensor(enc(train_df[etiqueta].values), dtype=torch.long)]
        vl_tensors = [t(x) for x in vl_arrs] + [torch.tensor(enc(val_df[etiqueta].values),   dtype=torch.long)]
    
        mk = lambda tens: DataLoader(TensorDataset(*tens), batch_size=batch_size, shuffle=False)
        return {"train_loader": mk(tr_tensors), "val_loader": mk(vl_tensors)}
    
    # Arquitectura basica de la red neuronal
    class MultiInputModel(nn.Module):
        def __init__(self, hidden_dim=128, dropout=0.3):
            super().__init__()
            sd = hidden_dim // 2
    
            def emb(i, o):   return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.BatchNorm1d(o), nn.Dropout(dropout), nn.Linear(o, o), nn.ReLU())
            def small(i, o): return nn.Sequential(nn.Linear(i, o), nn.ReLU(), nn.Linear(o, o), nn.ReLU())
    
            self.title_branch   = emb(768, hidden_dim)
            self.content_branch = emb(768, hidden_dim)
            self.verb_branch    = small(768, sd)
            self.context_branch = small(768, sd)
            self.meta_branch    = small(9, 32)
    
            cd        = hidden_dim * 2 + sd * 2
            self.gate = nn.Sequential(nn.Linear(cd, cd), nn.Sigmoid())
            self.head = nn.Sequential(
                nn.Linear(cd * 2 + 32, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128),         nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(128, 1))
    
        def forward(self, x_title, x_content, x_verb, x_context, x_meta):
            x_title, x_content = F.normalize(x_title, dim=1), F.normalize(x_content, dim=1)
            x_verb,  x_context = F.normalize(x_verb,  dim=1), F.normalize(x_context, dim=1)
            t, c = self.title_branch(x_title), self.content_branch(x_content)
            v, cx = self.verb_branch(x_verb),  self.context_branch(x_context)
            m        = self.meta_branch(x_meta)
            combined = torch.cat([t, c, v, cx], dim=1)
            gated    = combined * self.gate(combined)
            return self.head(torch.cat([combined, gated, m], dim=1))
    
    # Preparao los inpus y outputs para los modelos de ensemble
    def get_dataset(etiqueta):
        df_outputs = etiquetas[[etiqueta, "Tickers Mapeados", "Fila Noticia", "Date"]]
        df_final   = pd.concat([df_inputs, df_outputs.reset_index(drop=True)], axis=1).dropna()
        df_final   = df_final[df_final[etiqueta] != 0]
        ids        = df_final["Fila Noticia"].unique()
        n          = len(ids)
        train_ids  = ids[:int(n * 0.70)]
        val_ids    = ids[int(n * 0.70):int(n * 0.85)]
        train_df   = df_final[df_final["Fila Noticia"].isin(train_ids)]
        val_df     = df_final[df_final["Fila Noticia"].isin(val_ids)]
        return train_df, val_df
    
    # Reduzco los inputs de embedding con PCA
    def fit_pca(train_data, val_data, n=64):
        pca = PCA(n_components=n)
        return pca.fit_transform(train_data), pca.transform(val_data)
    
    # Adapto las etiquetas
    def encode(df, etiqueta):
        return np.vectorize(mapping.get)(df[etiqueta].astype(int))
    
    # Solo hago inferencia para los 50 mejores modelos segun el score global
    inference_dict = {}
    TOP_N_MODELS   = 50
    top_models_df  = total_metrics.sort_values(by="score_global", ascending=False).head(TOP_N_MODELS).reset_index(drop=True)
    
    # Iteracion para los top 50 modelos para calcular su inferencia
    for idx, model_row in top_models_df.iterrows():
        model_id = model_row["model_id"]
        modelo   = model_row["modelo"]
        etiqueta = model_row["ventana"]
    
        print(f"TOP: {idx} | MODELO: {model_id}")
        # Genero los inputs
        train_df, val_df = get_dataset(etiqueta)
        y_train = encode(train_df, etiqueta)
        y_val   = encode(val_df,   etiqueta)
    
        # Ensemble
        # Calculo las inferencias guarando probabilidad y etiquetas generadas para los modelos de ensemble
        if modelo in ["LGBM", "XGBoost"]:
            slices = [(0, 768), (768, 1536), (1536, 2304), (2304, 3072)]
            tr_pcas, vl_pcas = zip(*[fit_pca(train_df.iloc[:, a:b].values, val_df.iloc[:, a:b].values) for a, b in slices])
    
            X_train = np.concatenate([*tr_pcas, train_df.iloc[:, 3072:3081].values], axis=1)
            X_val   = np.concatenate([*vl_pcas, val_df.iloc[:, 3072:3081].values],   axis=1)
    
            arch = ensemble_architecture[ensemble_architecture["model_id"] == model_id].iloc[0]
    
            if modelo == "LGBM":
                model = LGBMClassifier(
                    verbosity=-1, random_state=42,
                    learning_rate=arch["learning_rate"], n_estimators=int(arch["n_estimators"]),
                    max_depth=int(arch["max_depth"]), num_leaves=int(arch["num_leaves"]),
                    min_child_samples=int(arch["min_child_samples"]),
                    subsample=arch["subsample"], colsample_bytree=arch["colsample_bytree"])
            else:
                model = XGBClassifier(
                    objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1,
                    learning_rate=arch["learning_rate"], n_estimators=int(arch["n_estimators"]),
                    max_depth=int(arch["max_depth"]), min_child_weight=int(arch["min_child_weight"]),
                    subsample=arch["subsample"], colsample_bytree=arch["colsample_bytree"])
    
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs > 0.5).astype(int)
    
        # Red neuronal
        # Calculo las inferencias guarando probabilidad y etiquetas generadas para la red neuronal
        else:
            data         = prepare_data(etiqueta=etiqueta, df_inputs=df_inputs, etiquetas=etiquetas)
            train_loader = data["train_loader"]
            val_loader   = data["val_loader"]
    
            arch  = NN_architecture[NN_architecture["model_id"] == model_id].iloc[0]
            model = MultiInputModel(hidden_dim=int(arch["hidden_dim"]), dropout=float(arch["dropout"])).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=float(arch["learning_rate"]))
            criterion = nn.BCEWithLogitsLoss()
    
            for _ in range(int(arch["epochs"])):
                model.train()
                for *xs, y_batch in train_loader:
                    xs      = [x.to(device) for x in xs]
                    y_batch = y_batch.view(-1, 1).float().to(device)
                    optimizer.zero_grad()
                    criterion(model(*xs), y_batch).backward()
                    optimizer.step()
    
            model.eval()
            probs, preds = [], []
            with torch.no_grad():
                for *xs, _ in val_loader:
                    xs          = [x.to(device) for x in xs]
                    batch_probs = torch.sigmoid(model(*xs)).cpu().numpy().flatten()
                    probs.extend(batch_probs)
                    preds.extend((batch_probs > 0.5).astype(int))
    
            probs = np.array(probs)
            preds = np.array(preds)
    
        # Guardo en un diccionario los datos de cada modelo del top 50 junto con un inferencia para pasarlo por el sistema trading
        inference_dict[idx] = {
            "model_id":    model_id,
            "modelo":      modelo,
            "ventana":     etiqueta,
            "inference_df": pd.DataFrame({
                "Tickers Mapeados": val_df["Tickers Mapeados"].values,
                "Fila Noticia":     val_df["Fila Noticia"].values,
                "Date":             val_df["Date"].values,
                "Prob_up":          probs,
                "Pred_label":       preds,
                "True_label":       y_val,
            })
        }
 
    # Guardo las metricas, arquitecutras e inferencias en las variables globales
    arquitectura_NN = NN_architecture
    arquitectura_ensemble = ensemble_architecture
    metricas_totales = total_metrics
    diccionario_inferencia_top50 = inference_dict
    
    

# Controlo la duracion de ejecucion
proceso1 = multiprocessing.Process(target=train_and_inference)
proceso1.start()
# Limite de maximo 2 minutos
proceso1.join(timeout=120)

# Si pasa de 2 minutos, dejo de ejecutar
if proceso1.is_alive():
    # Cierro por completo la ejecucion
    proceso1.terminate()
    proceso1.join() 
    print("\nSolo la carga de precios minuto a minuto tarda mas de 30 minutos, la busqueda de arquitecturas casi 1 hora y la inferencia 5 minutos")
    print("\nLas metricas del top50 ya lo tengo en mi tabla top50_metrics, las arquitecturas en NN_architecture y ensemble_architecture, y el diccionario en mi bucket top50-models-inferences")
else:
    print("El proceso de train e inference milagrosamente termino a tiempo")
    
    
    
    
    
    
    
    
