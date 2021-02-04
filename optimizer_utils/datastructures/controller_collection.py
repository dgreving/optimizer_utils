from collections import OrderedDict


class ControllerCollection(OrderedDict):
    """
    Wrapper class of OrderedDict, expecting instances of ParameterController,
    providing easy pickling and remote access from ParameterController
    instances.

    When added to a ControllerCollection, the save method a ParameterController
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._notify_controllers()
        self.parameter_list = None

    def _notify_controllers(self):
        for controller in self.values():
            if hasattr(controller, 'add_to_collection'):
                controller.add_to_collection(self)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            collection = pickle.load(f)
        return collection

    def create_parameter_list(self):
        lst = []
        for controller in self.values():
            for p in controller.plain_paras_to_list():
                if p not in lst:
                    lst.append(p)
        self.parameter_list = lst

    def __repr__(self):
        controllers_str = ', '.join(self.keys())
        return f"ControllerCollection containing: <{controllers_str}>"