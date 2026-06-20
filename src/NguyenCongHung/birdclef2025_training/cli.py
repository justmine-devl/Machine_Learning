from __future__ import annotations

import json

from .config import build_arg_parser, config_from_args
from .pipeline import BirdCLEFTrainingPipeline


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    print("Resolved config:")
    print(json.dumps(config.to_dict(), indent=2, default=str))
    outputs = BirdCLEFTrainingPipeline(config).run()
    print("Pipeline outputs:")
    print(json.dumps(outputs, indent=2, default=str))


if __name__ == "__main__":
    main()



