"""Export candidate plans without invoking the model or executing the query."""
import argparse
import json
from pathlib import Path
from .explorer import generate_candidate_plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sql-file', required=True)
    parser.add_argument('--query-id', required=True)
    parser.add_argument('--schema', default='public')
    parser.add_argument('--config', default='config/candidate_generation.json')
    parser.add_argument('--output-dir', default='generated_plans')
    args = parser.parse_args()
    records = generate_candidate_plans(args.schema, Path(args.sql_file).read_text(),
                                      args.query_id, config_path=args.config,
                                      opt_plan_path=args.output_dir)
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (args.query_id + '.candidates.json')).write_text(json.dumps(records, indent=2))
    print('Exported {} candidates'.format(len(records)))


if __name__ == '__main__':
    main()
