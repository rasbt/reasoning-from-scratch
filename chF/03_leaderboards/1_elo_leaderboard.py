# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch
import json
import argparse


def elo_ratings(vote_pairs, k_factor=32, initial_rating=1000):
    ratings = {
        model: initial_rating
        for pair in vote_pairs
        for model in pair
    }
    for winner, loser in vote_pairs:
        expected = 1.0 / (
            1.0 + 10 ** (
                (ratings[loser] - ratings[winner]) / 400.0
            )
        )
        ratings[winner] += k_factor * (1 - expected)
        ratings[loser] += k_factor * (0 - (1 - expected))
    return ratings


def main():
    parser = argparse.ArgumentParser(
        description="Compute Elo leaderboard."
    )
    parser.add_argument("--path", type=str, help="Path to votes JSON")
    parser.add_argument("--k", type=int, default=32,
                        help="Elo k-factor")
    parser.add_argument("--init", type=int, default=1000,
                        help="Initial rating")
    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        votes = json.load(f)

    ratings = elo_ratings(votes, args.k, args.init)
    leaderboard = sorted(ratings.items(),
                         key=lambda x: -x[1])

    print("\nLeaderboard (Elo) \n-----------------------")
    for i, (model, score) in enumerate(leaderboard, 1):
        print(f"{i:>2}. {model:<10} {score:7.1f}")
    print()


if __name__ == "__main__":
    main()
