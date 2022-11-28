import commander.COMMANDER_INTERFACE as commander

a, b, c = commander.get_random_img("./commander/arm.ndjson")
commander.img_to_strokes(a, b, c)
