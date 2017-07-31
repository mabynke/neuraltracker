'''
PROGRAM DESCRIPTION
This program converts information from a sqlite file to a json_file
'''

import sqlite3
import json
from PIL import Image


sqlite_filename = input('Enter the path of the sqlite file located in this folder (this folder should also contain all the image frames): ') # sherbrooke_gt.sqlite

conn = sqlite3.connect(sqlite_filename) # rouen_gt.sqlite
c = conn.cursor()

c.execute("SELECT object_id FROM objects")
sql_obj = c.fetchall()
number_of_objects = len(sql_obj)

c.execute("SELECT * FROM bounding_boxes")
sql_boundingbox = c.fetchall() # All data is in here, format : [(obj_id, frame_number, x1,y1,x2,y2),(obj_id, frame_number, ...), ...]

im = Image.open("00000001.jpg")
img_width = im.size[0] # TODO: get automatically
img_height = im.size[1]


print()
print('Image height: ' + str(img_height) + ', image width: ' + str(img_width))
print('SQlite filename: ' + str(sqlite_filename))
print('Number of different objects present in this image sequence: ' + str(number_of_objects) + '\n')

def json_format(my_obj_id):
	json_file = []

	for i in range(len(sql_boundingbox)):
		frameinfo = sql_boundingbox[i] #information regarding image 'i'
		object_id = frameinfo[0]

		while object_id == my_obj_id:
			frame_number = frameinfo[1]
			x1 = frameinfo[2]
			y1 = frameinfo[3]
			x2 = frameinfo[4]
			y2 = frameinfo[5]

			w = (x2-x1)/img_width
			h = (y2-y1)/img_height
			x = (x1*2)/img_width + w - 1
			y = (y1*2)/img_height + h - 1
			filename = "0"*(8-len(str(frame_number))) + str(frame_number) + ".jpg"

			dct = {"x":x, "y":y, "w": w, "h": h , "filename" : filename}
			json_file.append(dct)
			break

	json_file.append({"img_amount": len(json_file)})
	return json_file

for obj_nr in range(number_of_objects):
	json_file = json_format(obj_nr)

	print('Number of images with object nr' + str(obj_nr) + ' present: ' + str(len(json_file)) + "\n")

	with open("labels" + str(obj_nr) + ".json", "w") as labels_json:
		json.dump(json_file, labels_json) # dumps the content of json_file into labels_json.

print('finished! jsonfiles should now have appeared in this folder')