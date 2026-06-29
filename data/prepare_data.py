"""Mamba-3-Lite data prep: thin shim over the universal pipeline (GPT-2 BPE, vocab 50,257)."""
import argparse
import sys
from pathlib import Path

import yaml


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LLM_ROOT = _PROJECT_ROOT.parent.parent  # .../CoreProjects/
for _p in (_PROJECT_ROOT, _LLM_ROOT):
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)


MAMBA_TOKENIZER_NAME = "gpt2"
MAMBA_VOCAB_SIZE = 50_257
MAMBA_EOS_TOKEN_ID = 50_256
MAMBA_PAD_TOKEN_ID = 50_256


def _ensure_mamba_data_config(project_root: Path) -> Path:
    """Materialise a project-local data_config.yaml with Mamba-3-Lite's vocab."""
    from shared_data.config import UNIVERSAL_DATA_CONFIG_PATH
    from shared_data.common import load_yaml

    out_path = project_root / "data" / "data_config.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = load_yaml(UNIVERSAL_DATA_CONFIG_PATH)
    cfg["pipeline"]["tokenizer"]["name"] = MAMBA_TOKENIZER_NAME
    cfg["pipeline"]["tokenizer"]["vocab_size"] = MAMBA_VOCAB_SIZE
    cfg["pipeline"]["tokenizer"]["eos_token_id"] = MAMBA_EOS_TOKEN_ID
    cfg["pipeline"]["tokenizer"]["pad_token_id"] = MAMBA_PAD_TOKEN_ID
    cfg["_generator"] = "Mamba-3-Lite/data/prepare_data.py"
    cfg["_tokenizer_family"] = "gpt2"

    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _apply_mamba_defaults() -> Path:
    from shared_data.config import UNIVERSAL_TOTAL_TOKENS
    print(f"[data/mamba3] universal corpus: {UNIVERSAL_TOTAL_TOKENS:,} tokens")
    print(f"[data/mamba3] tokenizer: {MAMBA_TOKENIZER_NAME} "
          f"(vocab={MAMBA_VOCAB_SIZE:,}, EOS={MAMBA_EOS_TOKEN_ID})")
    print(f"[data/mamba3] shard size: 50,000,000 tokens (uint32)")
    return _ensure_mamba_data_config(Path(__file__).resolve().parents[1])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mamba-3-Lite data prep (delegates to universal pipeline)",
    )
    parser.add_argument("--stage", choices=["pretrain"], default="pretrain")
    parser.add_argument("--mixture", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument("--skip-pack", action="store_true")
    args = parser.parse_args()

    project_data_config = _apply_mamba_defaults()

    from shared_data.config import UNIVERSAL_MIXTURE_PATH
    from shared_data.prepare_data import run_pipeline

    return run_pipeline(
        mixture_path=Path(args.mixture) if args.mixture else UNIVERSAL_MIXTURE_PATH,
        data_config_path=Path(args.data_config) if args.data_config else project_data_config,
        source=args.source,
        skip_download=args.skip_download,
        skip_clean=args.skip_clean,
        skip_tokenize=args.skip_tokenize,
        skip_pack=args.skip_pack,
        data_root=Path(args.data_root) if args.data_root else None,
    )


if __name__ == "__main__":
    sys.exit(main())
