from infer_rvc_python import BaseLoader
import inspect

# List all public methods
methods = [m for m in dir(BaseLoader) if not m.startswith('_')]
print("Public methods:", methods)

# Show signature of key methods
for m in methods:
    try:
        sig = inspect.signature(getattr(BaseLoader, m))
        print(f"\n{m}{sig}")
    except Exception:
        pass
