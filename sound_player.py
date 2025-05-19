import pygame

class SoundPlayer:
    def __init__(self, sliding_open_sound, sliding_close_sound):
        pygame.mixer.init()
        self.sliding_open_sound = sliding_open_sound
        self.sliding_close_sound = sliding_close_sound
        self.last_state = None

    def play_transition_sound(self, new_state):
        if self.last_state == "open" and new_state == "sliding":
            self.play_sound_once(self.sliding_close_sound)
        elif self.last_state == "closed" and new_state == "sliding":
            self.play_sound_once(self.sliding_open_sound)
        elif new_state not in ["sliding"]:
            pygame.mixer.music.stop()
        self.last_state = new_state

    def play_sound_once(self, sound_file):
        pygame.mixer.music.load(sound_file)
        pygame.mixer.music.play(-1)
