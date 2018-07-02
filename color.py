import sty
from sty import fg, bg, ef, rs


def redness(text, color):
    return bg(255, color, color) + text + bg.rs

def blueness(text, color):
    return bg(color, color, 255) + text + bg.rs

def greeness(text, color):
    return bg(color, 255, color) + text + bg.rs




s = ""
for i in range(0, 256):
    s += redness("a", i)
    
print (s)
