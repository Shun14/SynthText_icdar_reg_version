import os


fontsList = os.listdir('./more_font')
f = open('fontlist.txt', 'w')
fonts = []
for j in fontsList:
  i ='more_font/'+ j
  fonts.append(i)
  fonts.append('\n')
print fonts

f.writelines(fonts)
f.close()     
