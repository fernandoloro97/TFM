def handler(event, context):
    print("Hola desde AWS Lambda y ECR")
    return {"message": "Ejecución exitosa"}

if __name__ == "__main__":
    print("Corriendo localmente")
