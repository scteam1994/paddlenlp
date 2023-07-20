from paddlenlp import Taskflow

docprompt = Taskflow("document_intelligence")
a = docprompt([{"doc": "https://bj.bcebos.com/paddlenlp/taskflow/document_intelligence/images/invoice.jpg",
                        "prompt": ["名称是什么?", "校验码是多少?"]}])
print()