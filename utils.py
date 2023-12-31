import argparse
from typing import Any

import yaml


class _Config:
    def __init__(self, **args):
        for key, value in args.items():
            if isinstance(value, dict):
                args[key] = _Config(**value)
        self.__dict__.update(args)

    def __repr__(self):
        return repr(self.__dict__)

    def _update(self, args: dict) -> None:
        if args == {}:
            return
        if args is None:
            return

        for key, value in args.items():
            if isinstance(value, dict):
                getattr(self, key)._update(value)
            else:
                setattr(self, key, value)

    def _return_dict(self) -> dict:
        return self.__return_dict(self)

    def __return_dict(self, config: Any) -> dict:
        state = {}
        for key, value in config.__dict__.items():
            if isinstance(value, _Config):
                state[key] = self.__return_dict(value)
            else:
                state[key] = value
        return state


class Config:
    def __init__(self, path: str, create_parser: bool = False) -> None:
        self.path = path
        self._load_yaml()

        if create_parser:
            self._create_parser()
            self._add_parser_args()
            self._parse_args()

    def update(self, args: dict) -> None:
        self.config._update(args)

    def save(self, path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.config._return_dict(), f)

    def __repr__(self) -> str:
        return repr(self.config)

    def __getattr__(self, key: str) -> Any:
        return getattr(self.config, key)

    def _load_yaml(self) -> None:
        with open(self.path, "r") as f:
            args = yaml.load(f, Loader=yaml.SafeLoader)
        self.config = _Config(**args)

    def _create_parser(self) -> None:
        self.parser = argparse.ArgumentParser()

    def _add_parser_args(self) -> None:
        self.__add_parser_args([], self.config)

    def __add_parser_args(self, parents: list, config: _Config) -> None:
        for key, value in config.__dict__.items():
            if isinstance(value, _Config):
                self.__add_parser_args(parents + [key], value)
            else:
                self.parser.add_argument(
                    "--" + ".".join(parents + [key]), default=value
                )

    def _create_recursive_dict(self, fields: list, value: Any) -> tuple:
        if len(fields) == 1:
            return fields, value

        return self._create_recursive_dict(fields[:-1], {fields[-1]: value})

    def _process_dict(self, args: dict) -> dict:
        processed_args = {}

        for key, value in args.items():
            if "." not in key:
                processed_args[key] = value

            else:
                fields = key.split(".")
                first_field = fields[0]
                if first_field not in processed_args:
                    processed_args[first_field] = {}
                fields, value = self._create_recursive_dict(fields[1:], value)
                processed_args[first_field][fields[0]] = value

        return processed_args

    def _parse_args(self) -> None:
        args = vars(self.parser.parse_args())
        self.update(self._process_dict(args))


import json


def read_json(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def write_json(path: str, data: dict, indent: int = 4) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def read_jsonl(path: str) -> list:
    data = []

    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def write_jsonl(path: str, data: list, indent: int = 4) -> None:
    with open(path, "w") as f:
        for line in data:
            json.dump(line, f, indent=indent)
            f.write("\n")


def read_txt(path: str) -> list:
    data = []

    with open(path, 'r') as f:
        for line in f:
            data.append(line.rstrip())

    return data

def write_txt(path: str, data: list) -> None:

    with open(path, 'w') as f:
        for line in data:
            f.write(line)
            f.write('\n')