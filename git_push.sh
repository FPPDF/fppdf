#!/bin/bash  


git remote add origin git@github.com:LucianHL/fixpar_nnpdf.git
git add *
git commit -m "Test upload"
git checkout -b main
git push origin main
