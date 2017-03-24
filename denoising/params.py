def get_params_grayscale(sigma):
    if sigma <= 15:
        return 3, 21, 0.4 * sigma
    if sigma <= 30:
        return 5, 21, 0.4 * sigma
    if sigma <= 45:
        return 7, 35, 0.35 * sigma


def get_params_color(sigma):
    if sigma <= 25:
        return 1, 10, 0.55 * sigma
    if sigma <= 55:
        return 2, 17, 0.4 * sigma
    if sigma <= 100:
        return 3, 17, 0.35 * sigma

    raise Exception('Sigma value too high')
