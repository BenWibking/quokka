import argparse
from pathlib import Path
import tarfile
from zipfile import ZipFile

from scanf import scanf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tarball_output')
    parser.add_argument('tex_files',nargs='*')
    args = parser.parse_args()
    
    files = []
    for tex_file in args.tex_files:
        tex_file_path = Path(tex_file)
        if tex_file_path.exists():
            files.append(str(tex_file_path))

        pdf_path = tex_file_path.with_suffix('.pdf')
        if pdf_path.exists() and str(pdf_path) not in files:
            files.append(str(pdf_path))

        # parse dep_file generated with \RequirePackage{snapshot}
        dep_file = tex_file_path.with_suffix('.dep')
        if dep_file.exists():
            with open(str(dep_file),'r') as f:
                for line in f:
                    if '*{file}' not in line:
                        continue

                    match = scanf('*{file} {%s}{0000/00/00 v0.0}', line, collapseWhitespace=True)
                    if match is None:
                        alt = scanf('*{file} {%s} {0000/00/00 v0.0}', line, collapseWhitespace=True)
                        if alt is None:
                            alt2 = scanf('*{file} {%s}{Graphic v0.0}', line, collapseWhitespace=True)
                            if alt2 is None:
                                alt3 = scanf('*{file} {%s} {Graphic v0.0}', line, collapseWhitespace=True)
                                if alt3 is None:
                                    continue
                                else:
                                    match = alt3
                            else:
                                match = alt2
                        else:
                            match = alt

                    filename, = match
                    path = Path(filename)
                    if path.suffix in ['.png','.pdf','.tex','.bbl','.cls'] and path.exists():
                        if str(path) not in files:
                            files.append(str(path))

    print("FILES IN TARBALL:\n")
    for myfile in files:
        print(myfile)

    # make tarball from files
    output_path = Path(args.tarball_output)
    if output_path.suffix == '.gz':
        with tarfile.open(args.tarball_output, 'w:gz', dereference=True) as tar:
            for this_file in files:
                tar.add(this_file)
    elif output_path.suffix == '.zip':
        with ZipFile(args.tarball_output, 'w') as myzip:
            for this_file in files:
                myzip.write(this_file)
    else:
        raise Exception('unrecognized output suffix')
