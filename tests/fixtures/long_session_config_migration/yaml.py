def safe_load(handle):
    data = {}
    for line in handle:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if value.isdigit():
            value = int(value)
        data[key.strip()] = value
    return data
