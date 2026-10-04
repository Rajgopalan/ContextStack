"""CLI: python -m src.cli <file1> [file2] [--provider mock] [--title ...]"""
import argparse
import json
import sys

from .generator import generate_aop, save_aop
from .models import aop_tokens, to_markdown


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--provider", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    if not 1 <= len(args.files) <= 2:
        print("Provide 1 or 2 files", file=sys.stderr)
        sys.exit(1)
    aop = generate_aop(args.files, title_override=args.title, provider_name=args.provider)
    path = save_aop(aop)
    print(to_markdown(aop))
    print(f"\n---\nJSON: {json.dumps(aop.model_dump(), indent=2)}", file=sys.stderr)
    print(f"Saved {path} + {path[:-5]}.md | ~{aop_tokens(aop)} tokens (md)", file=sys.stderr)


if __name__ == "__main__":
    main()
