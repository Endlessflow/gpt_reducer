# UTILITY FUNCTIONS
def append_to_file(file_name, string):
    with open(file_name, "a") as file:
        file.write(string)


def read_file(file_name):
    with open(file_name, "r") as file:
        return file.read()


def file_exists(file_name):
    try:
        with open(file_name, "r") as file:
            return True
    except:
        return False
