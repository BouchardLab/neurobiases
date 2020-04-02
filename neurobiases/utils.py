def read_attribute_dict(attributes):
    copy = {}
    for key, val in attributes.items():
        if val == '':
            copy[key] = None
        else:
            copy[key] = val
    return copy
