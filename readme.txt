Dependencies:
- OpenCL 1.1 or later
- GCC (any version) or MinGW
- Python 2.1 or later (only dependency for sequential implementation)
- Numpy (latest)
- Scipy (latest)
- PyOpenCL (any version)

OS tested on:
- Windows 8 and 10 (32/64 bits)
- Linux Elementary OS 0.3.1 and Ubuntu 15.10 (32/64 bits)

Limitations/known issues:
- Tested up to 40,000 parallel kernel instances

Software versions used:
- OpenCL 1.2
- Python 2.7.10
- PyOpenCL 2.4
- MinGW latest
- Numpy 1.10.4
- Scipy 0.17.0

Input files:
- input.txt
  - contains the following in order and delimited by empty lines:
  	- dimension (row, col) of trigger matrix
  	- trigger matrix (one line per row, elements delimited by whitespace)
  	- transition matrix (one line per row, elements delimited by whitespace)
  	- initial configuration vector (elements delimited by whitespaces)
- cand_apps.txt (optional)
  - contains the following in order and delimited by new lines:
  	- number of candidate application vectors
  	- candidate application vectors (delimited by new lines, elements delimited by whitespaces)

 Input format:
 - input.txt
   [row][col]

   [trigger matrix row 0]
   [trigger matrix row 1]
   [       ...          ]
   [       ...          ]
   [       ...          ]
   [trigger matrix row n]

   [transition matrix row 0]
   [transition matrix row 1]
   [          ...          ]
   [          ...          ]
   [          ...          ]
   [transition matrix row n]

   [initial configuration vector]

- cand_apps.txt
  [number of candidate application vectors]
  [candidate application vector 1]
  [             ...              ]
  [             ...              ]
  [candidate application vector n]

Reference of test inputs basis:
https://sites.google.com/a/dcs.upd.edu.ph/csp-proceedings/pcsc2011/Juayong.etal.pcsc2011.p133-139.pdf?attredirects=0&d=1