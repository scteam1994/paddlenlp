import re

import sys

from uvicorn.main import main

if __name__ == '__main__':
    arg = ["run_paddlenlp.py", 'server:app', '--workers', '1', '--host', '0.0.0.0', '--port', '8189']
    sys.argv = arg
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())
