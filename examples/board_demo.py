"""Tour of the board module. Run from the project folder:

    python examples/board_demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from railbaron import Board

board = Board.load()


def describe(route):
    legs = []
    for node, rr in zip(route.nodes[1:], route.railroads):
        if legs and legs[-1][0] == rr:
            legs[-1][1] = node
        else:
            legs.append([rr, node])
    return " | ".join(f"{rr} to {board.label(node)}" for rr, node in legs)


print("1. Shortest route")
route = board.shortest_route("New York", "Los Angeles")
print(f"   New York to Los Angeles: {route.distance} dots")
print(f"   {describe(route)}")

print("\n2. Route that avoids fees")
# Anything not listed belongs to the bank.
ownership = {"NYC": "Bob", "PA": "Bob", "B+O": "Bob", "C+O": "Ann"}
for step_cost in (1, 1000, 5000):
    route = board.cheapest_route("New York", "Chicago", ownership, "Ann", step_cost=step_cost)
    fee = board.estimate_fee(route, ownership, "Ann")
    print(f"   Ann, valuing a dot at ${step_cost}: {route.distance} dots, about ${fee} in fees")
    print(f"   {describe(route)}")

print("\n3. Where can a roll of 8 from Chicago end?")
ends = board.reachable("Chicago", 8)
cities = sorted(board.label(n) for n in ends if board.city_at(n))
print(f"   {len(ends)} dots, including the cities: {', '.join(cities)}")

print("\n4. Railroad values (busiest first)")
print(f"   {'line':8} {'price':>7} {'cities':>6} {'coverage':>9} {'traffic':>8} {'traffic/$10k':>13}")
for row in board.railroad_stats():
    print(
        f"   {row['railroad']:8} {row['price']:>7} {len(row['cities']):>6} "
        f"{row['coverage']:>9.1%} {row['traffic']:>8.2f} {row['traffic_per_10k']:>13.2f}"
    )

print("\n5. What should the owner of the PA buy next, for coverage?")
gains = sorted(((board.purchase_gain(["PA"], rr), rr) for rr in board.railroads), reverse=True)
for gain, rr in gains[:5]:
    print(f"   {rr:8} +{gain:.1%} of destinations for ${board.railroads[rr].price}")
