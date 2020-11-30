
# https://stackoverflow.com/questions/1057431/how-to-load-all-modules-in-a-folder

# from os.path import dirname, basename, isfile, join
# import glob
# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]


from os.path import splitext,join,basename,dirname
from glob import glob  
#get all *.py filenames in __file__'s folder that are not __file__.
files=[splitext(f)[0] for f in glob(join(dirname(__file__), '*.py')) 
       if f != basename(__file__)]
from importlib import import_module
for f in files:
    import_module("."+f,"VLAP") #import archive.f
    globals()[f]=getattr(globals[f],f) #assign the function f.f to variable f 
