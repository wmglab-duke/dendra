def validate(mechanism):
    for v in mechanism._write_ion.values():
        for i in v:
            if not callable(getattr(mechanism, i, None)):
                raise ValueError(f"current {v} not implemented")
    return True
