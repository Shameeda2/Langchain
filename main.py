from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("Hello from langchain!")
    print(os.environ.get("HUGGINGFACEHUB_API_TOKEN"))


if __name__ == "__main__":
    main()
