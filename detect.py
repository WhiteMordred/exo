import os
import ast
from collections import defaultdict

def list_imported_modules(project_dir):
    imported = set()
    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".py"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=file)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for alias in node.names:
                                    imported.add(alias.name.split('.')[0])
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imported.add(node.module.split('.')[0])
                    except Exception as e:
                        print(f"[⚠️] Erreur dans {file} : {e}")
    return sorted(imported)

if __name__ == "__main__":
    import sys
    project_path = sys.argv[1] if len(sys.argv) > 1 else "."
    modules = list_imported_modules(project_path)
    print("📦 Modules détectés dans le projet :")
    for m in modules:
        print(f" - {m}")