import re
import sys
from paddlenlp.cli import main
from paddlenlp import SimpleServer, Taskflow

cls = Taskflow("text_classification", task_path="/data/models/contract_first", is_static_model=True)
app = SimpleServer()
app.register_taskflow("taskflow/cls", cls)

if __name__ == '__main__':
    sys.argv = ['server', 'server:app', '--host', '0.0.0.0', '--port', '8189']

    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(main())