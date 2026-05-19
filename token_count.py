import tiktoken
import argparse

def count_tokens_in_file(file_path, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    tokens = encoding.encode(text)
    return len(tokens)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4")
    args = parser.parse_args()
    print(f"Token count: {count_tokens_in_file(args.file, args.model)}")

if __name__ == "__main__":
    main()