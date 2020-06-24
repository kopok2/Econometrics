from operator import itemgetter
import torch


def hellwig(corr, cov, n):
    res = []
    for i in range(1, 2 ** n):
        comb = "0" * (n - len((bin(i)[2:]))) + bin(i)[2:]
        print(comb)
        integral_capacity = 0
        for j in range(n):
            if comb[j] == "1":
                ss = 0
                for k in range(n):
                    if comb[k] == "1":
                        ss += abs(cov[j, k])
                h = (corr[j] ** 2) / ss
                integral_capacity += h
                print(h)
        print("Integral")
        print(integral_capacity)
        res.append((comb, integral_capacity))
    res.sort(key=itemgetter(1), reverse=True)
    return res[0]


if __name__ == '__main__':
    corr_vector = torch.tensor([.43, -.8, .18, .63])
    cov_matrix = torch.tensor([[1.0, -.64, .14, .41],
                               [-.64, 1.0, -.13, - .55],
                               [.14, -.13, 1.0, - .03],
                               [.41, -.55, -.03, 1.0]])
    print("x1 x2 x3 x4 | Rozwiązanie: (0 - bez zmiennej, 1 - umiesc zmienna)")
    print(hellwig(corr_vector, cov_matrix, corr_vector.shape[0]))

"""
0001 0.3968999981880188
0010 0.03240000084042549
0011 0.41679614782333374
0100 0.64000004529953
0101 0.6689678430557251 # Optymalne rozwiązanie - zmienne x2 oraz x4
0110 0.5950443148612976
0111 0.6600859761238098
1000 0.18490000069141388
1001 0.4126241207122803
1010 0.19061404466629028
1011 0.4226076304912567
1100 0.5029878616333008
1101 0.5849325656890869
1110 0.4909701645374298
1111 0.584661602973938
"""