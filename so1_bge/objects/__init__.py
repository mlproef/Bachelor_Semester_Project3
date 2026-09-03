"""So1 result objects. Profile pairs live in shared.profiles."""

from shared.profiles import UserPair

from so1_bge.objects.metrics import UserSimilarity, UserTokenChange

__all__ = ["UserPair", "UserSimilarity", "UserTokenChange"]
