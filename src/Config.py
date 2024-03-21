import os.path
import sys
from typing import Dict, Any, List


class ConfigLoader(type):
    def __new__(cls, clsname, bases, attrs, config_file: str):
        lc: Dict[str, Any] = {}
        gl: Dict[str, Any] = {}
        with open(config_file, 'r') as f:
            exec(f.read(), gl, lc)

        artifact_folder: str = lc['artifacts_folder'] if 'artifacts_folder' in lc else 'artifacts'

        if not os.path.exists(artifact_folder):
            os.makedirs(artifact_folder)

        for name, val in lc.items():
            if val is None:
                envname = name.upper()
                if envname not in os.environ:
                    continue
                    # print(f"Environment variable {envname} for config var {name} is not set", file=sys.stderr)
                else:
                    val = os.environ[envname]

            if type(val) == str and (name.endswith('path') or name.endswith('file')) and not os.path.isabs(val):
                val = os.path.join(artifact_folder, val)
            attrs[name] = val

        return super().__new__(cls, clsname, bases, attrs)


class Config(metaclass=ConfigLoader, config_file='config.py'):
    # Misc
    model_name: str
    faiss_use_gpu: bool
    show_plot: bool
    threshold: float

    # Files
    artifacts_folder: str
    embeddings_file: str
    mapping_file: str
    index_file: str
    centroid_file: str
    temp_index_file: str
    temp_mapping_file: str
    source_index_path: str  # Should be an absolute path

    # Wiki Articles
    page_names: List[str]
    unrelated_page_names: List[str]
    unrelated_page_names_2: List[str]

    @classmethod
    def artifact(cls, filename: str) -> str:
        return os.path.join(cls.artifacts_folder, filename)
