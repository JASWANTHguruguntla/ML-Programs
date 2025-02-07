# CSP Map colouring USing BACKTRACKING
def is_valid(map, region, color, color_assignment):
    for neighbor in map[region]:
        if neighbor in color_assignment and color_assignment[neighbor] == color:
            return False
    return True
def solve_map_coloring(map, regions, colors, color_assignment={}):
    if len(color_assignment) == len(regions):
        return color_assignment
    current_region = [r for r in regions if r not in color_assignment][0]
    for color in colors:
        if is_valid(map, current_region, color, color_assignment):
            color_assignment[current_region] = color
            result = solve_map_coloring(map, regions, colors, color_assignment)
            if result is not None:
                return result
            del color_assignment[current_region]
    return None
if __name__ == "__main__":
    map = {
        "WA": ["NT", "SA"],
        "NT": ["WA", "SA", "Q"],
        "SA": ["WA", "NT", "Q", "NSW", "V"],
        "Q": ["NT", "SA", "NSW"],
        "NSW": ["Q", "SA", "V"],
        "V": ["SA", "NSW"],
    }
    regions = list(map.keys())
    colors = ["Red", "Green", "Blue"]
    coloring = solve_map_coloring(map, regions, colors)
    if coloring:
        print("Valid coloring:")
        for region, color in coloring.items():
            print(f"{region}: {color}")
    else:
        print("No valid coloring found.")
