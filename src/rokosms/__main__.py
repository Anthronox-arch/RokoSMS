import sys
import json

from .predict import predict


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m rokosms \"your message here\"")
        sys.exit(1)

    text = sys.argv[1]
    result = predict(text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
