# Testing that my new slres behavior matches the original orbitNP_release_1.2.1

import os
import slres

def test_residuals():
    os.system("rm orbitNP_release_1.2.1/resids.dat && rm slres/resids.dat")

    os.system(
        "cd orbitNP_release_1.2.1 && echo '0' | python orbitNP.py -f ../data/blits_201206.frd -s 7841 -r -c ../data/blits_cpf_120626_6801.hts"
    )
    os.system("cd /home/liam/Documents/slr")

    slres.process_one('data/blits_201206.frd', 'data/blits_cpf_120626_6801.hts', '7841')

    # os.system(
    #     "cd slres && echo '0' | python slres.py -f ../data/blits_201206.frd -s 7841 -r -c ../data/blits_cpf_120626_6801.hts"
    # )
    # os.system("cd /home/liam/Documents/slr")

    with open("slres/resids.dat", "r") as f:
        f_slres = f.read()

    with open("orbitNP_release_1.2.1/resids.dat", "r") as f:
        f_orbitnp = f.read()

    assert f_slres == f_orbitnp, "Residuals don't match, something critical changed!"

if __name__ == "__main__":
    test_residuals()