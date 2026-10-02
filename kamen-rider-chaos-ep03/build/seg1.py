import sys
from PIL import Image
from rembg import remove, new_session
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGB')
if len(sys.argv) > 3: im = im.crop(tuple(int(v) for v in sys.argv[3].split(',')))
m = remove(im, session=new_session('birefnet-general'), only_mask=True, post_process_mask=False)
m.save(dst); print(dst, m.size)
