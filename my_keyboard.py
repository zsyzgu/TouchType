import keyboard

class MyKeyboard():
    def __init__(self):
        self.history = {}

    def _is_last_pressed(self, key):
        if key in self.history:
            return self.history[key]
        return False

    def is_pressed(self, key):
        self.history[key] = keyboard.is_pressed(key)
        return self.history[key]

    def is_pressed_down(self, key):
        last_pressed = self._is_last_pressed(key)
        curr_pressed = self.is_pressed(key)
        return not last_pressed and curr_pressed
    
    def is_pressed_up(self, key):
        last_pressed = self._is_last_pressed(key)
        curr_pressed = self.is_pressed(key)
        return last_pressed and not curr_pressed
    