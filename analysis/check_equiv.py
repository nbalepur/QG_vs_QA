import re

num_pattern = re.compile(r"\d+")

RANGE = 5

def is_equivalent(a: str, b: str, category: str) -> float:
    match category:
        case "num_prefix_single_lower":
            raise NotImplementedError("Not implemented")
        case "num_prefix_multiple_lower":
            raise NotImplementedError("Not implemented")
        case "num_prefix_upper":
            raise NotImplementedError("Not implemented")
        case "easy_fact":
            a = re.sub(r"[^A-Za-z0-9 \'\.]", " ", a)

            ref = a.lower()
            generated_entity = b.lower()

            # remove any special characters from ref

            if ref in generated_entity:
                return 1
            
            splitted_ref = a.split()
            capitalized = [s for s in splitted_ref if s[0].isupper() and len(s) > 3]

            return int(all(s in b for s in capitalized))

        case "hard_fact":
            a = re.sub(r"[^A-Za-z0-9 \'\.]", " ", a)

            ref = a.lower()
            generated_entity = b.lower()

            # remove any special characters from ref

            if ref in generated_entity:
                return 1
            
            splitted_ref = a.split()
            capitalized = [s for s in splitted_ref if s[0].isupper() and len(s) > 3]

            return int(any(s in b for s in capitalized))
        case "num":
            if a in b:
                return 1
            
            num = int(a)

            if any(str(i) in b for i in range(int(num) - 1, int(num) + 1)):
                return 0.5

            return 0
        case "num_suffix":
            raise NotImplementedError("Not implemented")
        case "num_text":
            num = re.findall(num_pattern, a)
            if len(num) == 0:
                raise ValueError(f"Number not found in {a}")
            """ if len(num) > 1:
                return -1 """
                #raise ValueError(f"Multiple numbers found in {a}")
            num = num[0]

            if num in b:
                return 1
            
            if any(str(i) in b for i in range(int(num) - 1, int(num) + 1)):
                return 1
            
            if any(str(i) in b for i in range(int(num) - RANGE, int(num) + RANGE)):
                return 0.5

            return 0
        case _:
            raise ValueError(f"Unknown category: {category}")
