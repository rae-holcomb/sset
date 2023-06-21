# This file is to save pieces of discarded code that is not currently being used, but which I might want to refer back to later.


# post_init for the TSGenerator class
    # def __post_init__(self) -> None:
    #     print('post init')
    #     pass
    #     """Loop through and convert all attributes to distributions.
    #     This isn't getting called for some reason and I can't figure out why."""
    #     print('__post_init__')
    #     for key, value in self.params.items():
    #         self.setattr(self, key, trc.convert_to_distribution(value))
    #     # pass


# post_init for the SineTSGenerator class
    # def __post_init__(self, A:stats.rv_continuous=1, B:stats.rv_continuous=1, C:stats.rv_continuous=0, D:stats.rv_continuous=0) -> None:
    #     """
    #     Sine function of the form y = A * sin(B(x+C) + D).

    #     Parameters
    #     ----------

    #     A: stats.Normal
    #     B:
    #     C:
    #     D:  
    #     """
    #     self.A = trc.convert_to_distribution(A)
    #     self.B = trc.convert_to_distribution(B)
    #     self.C = trc.convert_to_distribution(C)
    #     self.D = trc.convert_to_distribution(D)
