import os

where_am_i = os.popen('hostname').read()
print(where_am_i)