from .view_base import BaseView

class IdentityView(BaseView):
    def __init__(self):
        pass

    def view(self, im):
        return im

    def inverse_view(self, noise):
        return noise
