Image.open("path\\to\\image.jpg")
 >>> foo.size
  (200,374)
 # I downsize the image with an ANTIALIAS filter (gives the highest quality)
 >>> foo = foo.resize((160,300),Image.ANTIALIAS)
