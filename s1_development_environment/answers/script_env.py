#!/usr/bin/env python3
# import os
# print(os.environ["MY_VAR"])

from dotenv import load_dotenv
load_dotenv()
import os
print(os.environ["MY_OTHER_VAR"])

