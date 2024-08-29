from tkinter import Tk, Canvas, mainloop, Label
from PIL import ImageTk, Image
from trimesh.path.creation import rectangle

from lib.optimizer import exponential_decay_lr_

print(exponential_decay_lr_(0.3666666666666667,0.01,5*1e-6))
exit()

def clicked(event):
    button_number=(int(event.y/60)*9)+(1+int(event.x/60))
    print(f'You clicked button number {button_number}')

root=Tk()

img=ImageTk.PhotoImage(Image.open("Frame_0.ppm"))
panel=Label(root, image=img)
panel.pack(side="bottom", fill="both", expand="yes")

drawCanv=Canvas(width=541, height=301, bd=0)
drawCanv.bind('<Button-1>',clicked)

button_number = 1

for y in range(1,300,60):
    for x in range (1, 540,60):
        # rectangle = drawCanv.create_oval(x,y,x+60,y+60, outline='black')
        # drawCanv.create_text(x+25,y+30, text=button_number)
        button_number+=1

drawCanv.pack()
mainloop()