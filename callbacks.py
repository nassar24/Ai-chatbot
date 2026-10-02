# Callbacks for git-filter-repo

import re

REPLACEMENTS = [
    (b'PixNoise', b'Apex Creative'),
    (b'pixnoise', b'apexcreative'),
    (b'PIXNOISE', b'APEX CREATIVE'),
    (b'pix_noise', b'apex_creative'),
    (b'info@pixnoise.com', b'info@apexcreative.example'),
    (b'hr@pixnoise.com', b'hr@apexcreative.example'),
    (b'pixnoise.com', b'apexcreative.example'),
    (b'bot@pixnoise.com', b'bot@apexcreative.example'),
    (b'sales@pixnoise.com', b'sales@apexcreative.example'),
    (b'ai_agent@pixnoise.com', b'ai_agent@apexcreative.example'),
    (b'01157320969', b'+1-555-0100'),
    (b'Stanley, Alexandria, Egypt', b'123 Business Ave, Suite 400, San Francisco, CA 94102'),
    (b'https://web.facebook.com/pixnoise', b'https://facebook.com/apexcreative'),
    (b'https://www.instagram.com/pixnoiseagency/', b'https://instagram.com/apexcreative'),
    (b'https://www.linkedin.com/company/pixnoise', b'https://linkedin.com/company/apexcreative'),
    (b'https://www.behance.net/pixnoisemedia', b'https://behance.net/apexcreative'),
    (b'https://www.tiktok.com/@pixnoise_marketing', b'https://tiktok.com/@apexcreative'),
    (b'Dr. Muhammad Abaza', b'Dr. Alex Chen'),
    (b'Jana Mohamed', b'Jordan Smith'),
    (b'Basemmah', b'Casey Johnson'),
    (b'Adham', b'Taylor Williams'),
    (b'Yehia', b'Morgan Davis'),
    (b'Mahenour', b'Riley Brown'),
    (b'Roaya', b'Avery Wilson'),
    (b'Mohamed El Sakka', b'Quinn Taylor'),
    (b'Ahmed Nassar', b'Cameron Anderson'),
    (b'Abdelaziz', b'Dakota Martinez'),
    (b'\xd8\xaf. \xd9\x85\xd8\xad\xd9\x85\xd8\xaf \xd8\xa3\xd8\xa8\xd8\xa7\xd8\xb6\xd8\xa9', b'\xd8\xaf. \xd8\xa7\xd9\x84\xd9\x8a\xd9\x83\xd8\xb3 \xd8\xaa\xd8\xb4\xd9\x8a\xd9\x86'),
    (b'\xd8\xac\xd9\x86\xd9\x89 \xd9\x85\xd8\xad\xd9\x85\xd8\xaf', b'\xd8\xac\xd9\x88\xd8\xb1\xd8\xaf\xd8\xa7\xd9\x86 \xd8\xb3\xd9\x85\xd9\x8a\xd8\xab'),
    (b'\xd8\xa8\xd8\xa7\xd8\xb3\xd9\x85\xd9\x87', b'\xd9\x83\xd9\x8a\xd8\xb3\xd9\x8a \xd8\xac\xd9\x88\xd9\x87\xd9\x86\xd8\xb3\xd9\x88\xd9\x86'),
    (b'\xd8\xa3\xd8\xaf\xd9\x87\xd9\x85', b'\xd8\xaa\xd8\xa7\xd9\x8a\xd9\x84\xd9\x88\xd8\xb1 \xd9\x88\xd9\x8a\xd9\x84\xd9\x8a\xd8\xa7\xd9\x85\xd8\xb3'),
    (b'\xd9\x8a\xd8\xa7\xd8\xa7\xd8\xa8', b'\xd8\xaa\xd8\xa7\xd9\x8a\xd9\x84\xd9\x88\xd8\xb1 \xd9\x88\xd9\x8a\xd9\x84\xd9\x8a\xd8\xa7\xd9\x85\xd8\xb3'),
    (b'\xd9\x85\xd8\xa7\xd9\x87\xd9\x86\xd9\x88\xd8\xb1', b'\xd8\xb1\xd8\xa7\xd9\x8a\xd9\x84\xd9\x8a \xd8\xa8\xd8\xb1\xd8\xa7\xd9\x88\xd9\x86'),
    (b'\xd8\xb1\xd8\xa4\xd9\x8a\xd8\xa7', b'\xd8\xa5\xd9\x81\xd8\xb1\xd9\x8a \xd9\x88\xd9\x8a\xd9\x84\xd8\xb3\xd9\x88\xd9\x86'),
    (b'\xd9\x85\xd8\xad\xd9\x85\xd8\xaf \xd8\xa7\xd9\x84\xd8\xb3\xd9\x82\xd8\xa7', b'\xd9\x83\xd9\x88\xd9\x8a\xd9\x86 \xd8\xaa\xd8\xa7\xd9\x8a\xd9\x84\xd9\x88\xd8\xb1'),
    (b'\xd8\xa3\xd8\xad\xd9\x85\xd8\xaf \xd9\x86\xd8\xb5\xd8\xa7\xd8\xb1', b'\xd9\x83\xd8\xa7\xd9\x85\xd9\x8a\xd8\xb1\xd9\x88\xd9\x86 \xd8\xa3\xd9\x86\xd8\xaf\xd8\xb1\xd8\xb3\xd9\x88\xd9\x86'),
    (b'\xd8\xb9\xd8\xa8\xd8\xa7\xd8\xa9 \xd8\xa7\xd9\x84\xd8\xb9\xd8\xb2\xd9\x8a\xd8\xb2', b'\xd8\xaf\xd8\xa7\xd9\x83\xd9\x88\xd8\xaa\xd8\xa7 \xd9\x85\xd8\xa7\xd8\xb1\xd8\xaa\xd9\x8a\xd9\x86\xd8\xb2'),
]

def apply_replacements(data):
    for old, new in REPLACEMENTS:
        data = data.replace(old, new)
    return data

# Message callback (commit messages)
def message_callback(message):
    return apply_replacements(message)

# Blob callback (file contents)
def blob_callback(blob):
    blob.data = apply_replacements(blob.data)

# Name callback (author/committer names)
def name_callback(name):
    return apply_replacements(name)

# Email callback
def email_callback(email):
    return apply_replacements(email)

# Refname callback (branch/tag names)
def refname_callback(refname):
    return apply_replacements(refname)

# Filename callback (file paths)
def filename_callback(filename):
    new_name = apply_replacements(filename)
    return new_name