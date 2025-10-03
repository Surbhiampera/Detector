# +++++++++++ CUSTOM WSGI +++++++++++
# If you have a WSGI file that you want to serve using PythonAnywhere, perhaps
# in your home directory under version control, then use something like this:
#
import sys

# Add your project directory to the Python path
path = '/home/ampara/Downloads/Signal Detector'
if path not in sys.path:
    sys.path.append(path)

# Import your FastAPI app
from app_v2 import app

# For PythonAnywhere with FastAPI, we need to use mangum
# which provides ASGI-to-WSGI compatibility
try:
    from mangum import Mangum
    # Create a WSGI application by wrapping the ASGI app
    application = Mangum(app)
except ImportError:
    # Fallback: if mangum is not available, we'll create a simple WSGI wrapper
    # This is a basic implementation that may not work perfectly with all FastAPI features
    def application(environ, start_response):
        # This is a minimal WSGI wrapper - for production use, install mangum
        status = '200 OK'
        headers = [('Content-type', 'text/plain')]
        start_response(status, headers)
        return [b'FastAPI app requires mangum for WSGI compatibility. Please install: pip install mangum']
