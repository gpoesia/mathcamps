const standardData = {
  "K.CC.C.7": "Compare nums. btwn. 1 and 10",
  "K.OA.A.4": "Adding to equal 10",
  "K.OA.A.5": "Add/sub within 5",
  "K.NBT.A.1": "Decompose into 10s",
  "1.OA.A.1": "Add/sub within 20",
  "1.OA.A.2": "Add three nums within 20",
  "1.OA.D.8": "Determine unknowns in add/sub probs",
  "2.OA.A.1": "Add/sub within 100",
  "2.NBT.B.5": "Add/sub within 100",
  "2.NBT.B.6": "Add four 2 digit nums",
  "2.NBT.B.7": "Add/sub within 100",
  "2.MD.B.5": "Add/sub within 100",
  "2.MD.C.8": "Add/sub within 100",
  "3.OA.A.3": "Mul/div within 100",
  "3.OA.A.4": "Determine unknowns in mul/div probs",
  "3.OA.C.7": "Mul/div within 100",
  "3.OA.D.8": "Two-step word probs",
  "3.MD.D.8-triangle": "Triangle perimeter",
  "3.MD.D.8-quadrilateral": "Quadrilateral perimeter",
  "3.MD.D.8-polygon": "Polygon perimeter",
  "3.NBT.A.2": "Add/sub within 1000",
  "4.OA.A.3": "Multistep word probs",
  "4.OA.B.4": "Factor pairs within 100",
  "4.NBT.B.4": "Add/sub multi-digit nums",
  "4.NBT.B.5": "4 digit * 1 digit mult",
  "4.NBT.B.6": "4 digit / 1 digit div",
  "4.NF.A.2": "Compare two fractions",
  "4.MD.A.2-decimal": "Word problems with decimals",
  "4.MD.A.2-fraction": "Word problems with fractions",
  "4.MD.A.3": "Rectangle area and perimeter",
  "5.OA.A.1": "Evaluating with parentheses",
  "5.NBT.B.5": "Mult multi-digit nums",
  "5.NBT.B.6": "4 digit / 2 digit div",
  "5.NBT.B.7": "Add/sub/mult/div decimals",
  "5.NF.A.1": "Add/sub fractions",
  "5.NF.A.2": "Add/sub fraction word problems",
  "5.NF.B.4": "Mult fractions",
  "6.NS.B.2": "Multi-digit div",
  "6.NS.B.3": "Add/sub/mult/div decimals",
  "6.EE.A.1": "Evaluate exponents",
  "6.EE.B.7": "Solve one-variable systems",
  "7.NS.A.1-fraction": "Add/sub with fractions",
  "7.NS.A.1-decimal": "Add/sub with decimals",
  "7.NS.A.2": "Mult/div with fractions",
  "7.NS.A.3-fraction": "Add/sub/mult/div with fractions",
  "7.NS.A.3-decimal": "Add/sub/mult/div with decimals",
  "8.EE.A.2": "Calculate square and cube roots",
  "8.EE.C.7": "Solve one-variable systems",
  "8.EE.C.8": "Solve two-variable systems",

}
const modelData = {
  'claude-3-opus-20240229': {
    name: 'Claude-3 Opus',
    skills: {
      "K.CC.C.7": "1.00",
      "K.NBT.A.1": "0.95",
      "K.OA.A.4": "0.99",
      "K.OA.A.5": "0.96",
      "1.OA.A.1": "0.98",
      "1.OA.A.2": "0.98",
      "1.OA.D.8": "1.00",
      "2.MD.B.5": "0.95",
      "2.MD.C.8": "0.99",
      "2.NBT.B.5": "0.97",
      "2.NBT.B.6": "0.90",
      "2.NBT.B.7": "0.99",
      "2.OA.A.1": "0.98",
      "3.MD.D.8-polygon": "0.97",
      "3.MD.D.8-quadrilateral": "0.99",
      "3.MD.D.8-triangle": "1.00",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.97",
      "3.OA.A.4": "0.99",
      "3.OA.C.7": "0.97",
      "3.OA.D.8": "0.94",
      "4.MD.A.2-decimal": "0.95",
      "4.MD.A.2-fraction": "0.41",
      "4.MD.A.3": "0.99",
      "4.NBT.B.4": "0.98",
      "4.NBT.B.5": "1.00",
      "4.NBT.B.6": "0.94",
      "4.NF.A.2": "1.00",
      "4.OA.A.3": "0.77",
      "4.OA.B.4": "0.99",
      "5.NBT.B.5": "0.95",
      "5.NBT.B.6": "0.99",
      "5.NBT.B.7": "0.89",
      "5.NF.A.1": "0.50",
      "5.NF.A.2": "0.70",
      "5.NF.B.4": "0.71",
      "5.OA.A.1": "0.95",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "1.00",
      "6.NS.B.2": "0.99",
      "6.NS.B.3": "0.81",
      "7.NS.A.1-decimal": "0.99",
      "7.NS.A.1-fraction": "0.47",
      "7.NS.A.2": "0.81",
      "7.NS.A.3-decimal": "0.91",
      "7.NS.A.3-fraction": "0.35",
      "8.EE.A.2": "0.96",
      "8.EE.C.7": "0.27",
      "8.EE.C.8": "0.01",
    }
  },
  'claude-3-sonnet-20240229': {
    name: 'Claude-3 Sonnet',
    skills: {
      "K.CC.C.7": "1.00",
      "K.NBT.A.1": "0.94",
      "K.OA.A.4": "0.98",
      "K.OA.A.5": "0.94",
      "1.OA.A.1": "0.97",
      "1.OA.A.2": "0.98",
      "1.OA.D.8": "1.00",
      "2.MD.B.5": "0.97",
      "2.MD.C.8": "0.99",
      "2.NBT.B.5": "0.96",
      "2.NBT.B.6": "0.92",
      "2.NBT.B.7": "0.99",
      "2.OA.A.1": "0.98",
      "3.MD.D.8-polygon": "0.99",
      "3.MD.D.8-quadrilateral": "1.00",
      "3.MD.D.8-triangle": "1.00",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.94",
      "3.OA.A.4": "0.97",
      "3.OA.C.7": "0.98",
      "3.OA.D.8": "0.95",
      "4.MD.A.2-decimal": "0.89",
      "4.MD.A.2-fraction": "0.35",
      "4.MD.A.3": "1.00",
      "4.NBT.B.4": "0.98",
      "4.NBT.B.5": "0.99",
      "4.NBT.B.6": "0.88",
      "4.NF.A.2": "0.99",
      "4.OA.A.3": "0.79",
      "4.OA.B.4": "0.97",
      "5.NBT.B.5": "0.97",
      "5.NBT.B.6": "0.96",
      "5.NBT.B.7": "0.84",
      "5.NF.A.1": "0.32",
      "5.NF.A.2": "0.34",
      "5.NF.B.4": "0.55",
      "5.OA.A.1": "0.93",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "1.00",
      "6.NS.B.2": "0.94",
      "6.NS.B.3": "0.79",
      "7.NS.A.1-decimal": "1.00",
      "7.NS.A.1-fraction": "0.32",
      "7.NS.A.2": "0.59",
      "7.NS.A.3-decimal": "0.92",
      "7.NS.A.3-fraction": "0.18",
      "8.EE.A.2": "0.96",
      "8.EE.C.7": "0.23",
      "8.EE.C.8": "0.00",
    }
  },
  'claude-3-haiku-20240307': {
    name: 'Claude-3 Haiku',
    skills: {
      "K.CC.C.7": "0.99",
      "K.NBT.A.1": "0.94",
      "K.OA.A.4": "0.98",
      "K.OA.A.5": "0.96",
      "1.OA.A.1": "0.98",
      "1.OA.A.2": "0.98",
      "1.OA.D.8": "0.99",
      "2.MD.B.5": "0.97",
      "2.MD.C.8": "0.99",
      "2.NBT.B.5": "0.97",
      "2.NBT.B.6": "0.94",
      "2.NBT.B.7": "0.98",
      "2.OA.A.1": "0.97",
      "3.MD.D.8-polygon": "0.96",
      "3.MD.D.8-quadrilateral": "0.99",
      "3.MD.D.8-triangle": "1.00",
      "3.NBT.A.2": "0.99",
      "3.OA.A.3": "0.97",
      "3.OA.A.4": "1.00",
      "3.OA.C.7": "0.98",
      "3.OA.D.8": "0.94",
      "4.MD.A.2-decimal": "0.93",
      "4.MD.A.2-fraction": "0.37",
      "4.MD.A.3": "0.99",
      "4.NBT.B.4": "0.98",
      "4.NBT.B.5": "1.00",
      "4.NBT.B.6": "0.82",
      "4.NF.A.2": "0.97",
      "4.OA.A.3": "0.78",
      "4.OA.B.4": "1.00",
      "5.NBT.B.5": "0.93",
      "5.NBT.B.6": "0.98",
      "5.NBT.B.7": "0.73",
      "5.NF.A.1": "0.14",
      "5.NF.A.2": "0.39",
      "5.NF.B.4": "0.58",
      "5.OA.A.1": "0.91",
      "6.EE.A.1": "0.99",
      "6.EE.B.7": "0.98",
      "6.NS.B.2": "0.93",
      "6.NS.B.3": "0.78",
      "7.NS.A.1-decimal": "0.98",
      "7.NS.A.1-fraction": "0.22",
      "7.NS.A.2": "0.48",
      "7.NS.A.3-decimal": "0.90",
      "7.NS.A.3-fraction": "0.15",
      "8.EE.A.2": "0.96",
      "8.EE.C.7": "0.29",
      "8.EE.C.8": "0.00",
    }
  },
  'deepseek-ai/deepseek-coder-33b-instruct': {
    name: 'DeepSeek Coder 33B',
    skills: {
      "K.CC.C.7": "0.28",
      "K.NBT.A.1": "0.65",
      "K.OA.A.4": "0.73",
      "K.OA.A.5": "0.47",
      "1.OA.A.1": "0.46",
      "1.OA.A.2": "0.13",
      "1.OA.D.8": "0.54",
      "2.MD.B.5": "0.41",
      "2.MD.C.8": "0.46",
      "2.NBT.B.5": "0.34",
      "2.NBT.B.6": "0.13",
      "2.NBT.B.7": "0.34",
      "2.OA.A.1": "0.59",
      "3.MD.D.8-polygon": "0.01",
      "3.MD.D.8-quadrilateral": "0.17",
      "3.MD.D.8-triangle": "0.03",
      "3.NBT.A.2": "0.33",
      "3.OA.A.3": "0.54",
      "3.OA.A.4": "0.58",
      "3.OA.C.7": "0.52",
      "3.OA.D.8": "0.41",
      "4.MD.A.2-decimal": "0.31",
      "4.MD.A.2-fraction": "0.14",
      "4.MD.A.3": "0.23",
      "4.NBT.B.4": "0.23",
      "4.NBT.B.5": "0.27",
      "4.NBT.B.6": "0.23",
      "4.NF.A.2": "0.09",
      "4.OA.A.3": "0.21",
      "4.OA.B.4": "0.01",
      "5.NBT.B.5": "0.14",
      "5.NBT.B.6": "0.19",
      "5.NBT.B.7": "0.11",
      "5.NF.A.1": "0.00",
      "5.NF.A.2": "0.04",
      "5.NF.B.4": "0.03",
      "5.OA.A.1": "0.20",
      "6.EE.A.1": "0.01",
      "6.EE.B.7": "0.34",
      "6.NS.B.2": "0.13",
      "6.NS.B.3": "0.14",
      "7.NS.A.1-decimal": "0.12",
      "7.NS.A.1-fraction": "0.00",
      "7.NS.A.2": "0.00",
      "7.NS.A.3-decimal": "0.37",
      "7.NS.A.3-fraction": "0.02",
      "8.EE.A.2": "0.06",
      "8.EE.C.7": "0.00",
      "8.EE.C.8": "0.00",
    }
  },
  'deepseek-ai/deepseek-llm-67b-chat': {
    name: 'DeepSeek 67B',
    skills: {
      "K.CC.C.7": "0.86",
      "K.NBT.A.1": "0.97",
      "K.OA.A.4": "0.98",
      "K.OA.A.5": "0.93",
      "1.OA.A.1": "0.99",
      "1.OA.A.2": "0.98",
      "1.OA.D.8": "0.97",
      "2.MD.B.5": "0.98",
      "2.MD.C.8": "0.97",
      "2.NBT.B.5": "0.94",
      "2.NBT.B.6": "0.91",
      "2.NBT.B.7": "0.98",
      "2.OA.A.1": "0.87",
      "3.MD.D.8-polygon": "0.72",
      "3.MD.D.8-quadrilateral": "0.84",
      "3.MD.D.8-triangle": "0.40",
      "3.NBT.A.2": "0.96",
      "3.OA.A.3": "0.92",
      "3.OA.A.4": "0.96",
      "3.OA.C.7": "0.94",
      "3.OA.D.8": "0.96",
      "4.MD.A.2-decimal": "0.79",
      "4.MD.A.2-fraction": "0.33",
      "4.MD.A.3": "0.91",
      "4.NBT.B.4": "0.93",
      "4.NBT.B.5": "0.83",
      "4.NBT.B.6": "0.58",
      "4.NF.A.2": "0.40",
      "4.OA.A.3": "0.62",
      "4.OA.B.4": "0.95",
      "5.NBT.B.5": "0.46",
      "5.NBT.B.6": "0.83",
      "5.NBT.B.7": "0.58",
      "5.NF.A.1": "0.07",
      "5.NF.A.2": "0.31",
      "5.NF.B.4": "0.44",
      "5.OA.A.1": "0.74",
      "6.EE.A.1": "0.93",
      "6.EE.B.7": "0.93",
      "6.NS.B.2": "0.70",
      "6.NS.B.3": "0.53",
      "7.NS.A.1-decimal": "0.66",
      "7.NS.A.1-fraction": "0.05",
      "7.NS.A.2": "0.46",
      "7.NS.A.3-decimal": "0.84",
      "7.NS.A.3-fraction": "0.13",
      "8.EE.A.2": "0.86",
      "8.EE.C.7": "0.14",
      "8.EE.C.8": "0.00",
    }
  },
  'EleutherAI/llemma_7b': {
    name: 'LLemma 7B',
    skills: {
      "K.CC.C.7": "0.74",
      "K.NBT.A.1": "0.83",
      "K.OA.A.4": "0.96",
      "K.OA.A.5": "0.41",
      "1.OA.A.1": "0.93",
      "1.OA.A.2": "0.87",
      "1.OA.D.8": "0.86",
      "2.MD.B.5": "0.89",
      "2.MD.C.8": "0.85",
      "2.NBT.B.5": "0.90",
      "2.NBT.B.6": "0.77",
      "2.NBT.B.7": "0.87",
      "2.OA.A.1": "0.77",
      "3.MD.D.8-polygon": "0.56",
      "3.MD.D.8-quadrilateral": "0.59",
      "3.MD.D.8-triangle": "0.71",
      "3.NBT.A.2": "0.92",
      "3.OA.A.3": "0.93",
      "3.OA.A.4": "0.89",
      "3.OA.C.7": "0.86",
      "3.OA.D.8": "0.76",
      "4.MD.A.2-decimal": "0.56",
      "4.MD.A.2-fraction": "0.25",
      "4.MD.A.3": "0.43",
      "4.NBT.B.4": "0.82",
      "4.NBT.B.5": "0.45",
      "4.NBT.B.6": "0.16",
      "4.NF.A.2": "0.65",
      "4.OA.A.3": "0.29",
      "4.OA.B.4": "0.53",
      "5.NBT.B.5": "0.19",
      "5.NBT.B.6": "0.73",
      "5.NBT.B.7": "0.61",
      "5.NF.A.1": "0.03",
      "5.NF.A.2": "0.09",
      "5.NF.B.4": "0.44",
      "5.OA.A.1": "0.48",
      "6.EE.A.1": "0.96",
      "6.EE.B.7": "0.75",
      "6.NS.B.2": "0.42",
      "6.NS.B.3": "0.43",
      "7.NS.A.1-decimal": "0.86",
      "7.NS.A.1-fraction": "0.01",
      "7.NS.A.2": "0.24",
      "7.NS.A.3-decimal": "0.64",
      "7.NS.A.3-fraction": "0.05",
      "8.EE.A.2": "0.65",
      "8.EE.C.7": "0.22",
      "8.EE.C.8": "0.00",
    }
  },
  'gemini-1.5-pro-001': {
    name: 'Gemini-1.5 Pro',
    skills: {
      "K.CC.C.7": "0.96",
      "K.NBT.A.1": "0.94",
      "K.OA.A.4": "0.92",
      "K.OA.A.5": "0.97",
      "1.OA.A.1": "0.98",
      "1.OA.A.2": "0.98",
      "1.OA.D.8": "0.98",
      "2.MD.B.5": "0.95",
      "2.MD.C.8": "0.98",
      "2.NBT.B.5": "0.98",
      "2.NBT.B.6": "0.94",
      "2.NBT.B.7": "1.00",
      "2.OA.A.1": "0.97",
      "3.MD.D.8-polygon": "0.92",
      "3.MD.D.8-quadrilateral": "0.98",
      "3.MD.D.8-triangle": "1.00",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.93",
      "3.OA.A.4": "1.00",
      "3.OA.C.7": "0.97",
      "3.OA.D.8": "0.97",
      "4.MD.A.2-decimal": "0.89",
      "4.MD.A.2-fraction": "0.53",
      "4.MD.A.3": "0.99",
      "4.NBT.B.4": "0.99",
      "4.NBT.B.5": "0.99",
      "4.NBT.B.6": "0.88",
      "4.NF.A.2": "0.99",
      "4.OA.A.3": "0.78",
      "4.OA.B.4": "0.94",
      "5.NBT.B.5": "0.93",
      "5.NBT.B.6": "1.00",
      "5.NBT.B.7": "0.79",
      "5.NF.A.1": "0.50",
      "5.NF.A.2": "0.79",
      "5.NF.B.4": "0.79",
      "5.OA.A.1": "0.95",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "0.97",
      "6.NS.B.2": "0.98",
      "6.NS.B.3": "0.76",
      "7.NS.A.1-decimal": "0.98",
      "7.NS.A.1-fraction": "0.58",
      "7.NS.A.2": "0.84",
      "7.NS.A.3-decimal": "0.90",
      "7.NS.A.3-fraction": "0.50",
      "8.EE.A.2": "0.96",
      "8.EE.C.7": "0.40",
      "8.EE.C.8": "0.02",
    }
  },
  'gemini-1.5-flash-001': {
    name: 'Gemini-1.5 Flash',
    skills: {
      "K.CC.C.7": "0.97",
      "K.NBT.A.1": "0.98",
      "K.OA.A.4": "0.98",
      "K.OA.A.5": "0.97",
      "1.OA.A.1": "0.99",
      "1.OA.A.2": "0.96",
      "1.OA.D.8": "1.00",
      "2.MD.B.5": "0.96",
      "2.MD.C.8": "0.98",
      "2.NBT.B.5": "0.99",
      "2.NBT.B.6": "0.91",
      "2.NBT.B.7": "0.98",
      "2.OA.A.1": "0.99",
      "3.MD.D.8-polygon": "0.92",
      "3.MD.D.8-quadrilateral": "0.98",
      "3.MD.D.8-triangle": "1.00",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.96",
      "3.OA.A.4": "0.99",
      "3.OA.C.7": "0.99",
      "3.OA.D.8": "0.97",
      "4.MD.A.2-decimal": "0.93",
      "4.MD.A.2-fraction": "0.53",
      "4.MD.A.3": "0.99",
      "4.NBT.B.4": "0.96",
      "4.NBT.B.5": "0.99",
      "4.NBT.B.6": "0.88",
      "4.NF.A.2": "0.71",
      "4.OA.A.3": "0.80",
      "4.OA.B.4": "0.11",
      "5.NBT.B.5": "0.73",
      "5.NBT.B.6": "1.00",
      "5.NBT.B.7": "0.70",
      "5.NF.A.1": "0.48",
      "5.NF.A.2": "0.81",
      "5.NF.B.4": "0.76",
      "5.OA.A.1": "0.92",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "0.96",
      "6.NS.B.2": "0.96",
      "6.NS.B.3": "0.66",
      "7.NS.A.1-decimal": "0.99",
      "7.NS.A.1-fraction": "0.65",
      "7.NS.A.2": "0.86",
      "7.NS.A.3-decimal": "0.93",
      "7.NS.A.3-fraction": "0.59",
      "8.EE.A.2": "0.96",
      "8.EE.C.7": "0.51",
      "8.EE.C.8": "0.00",
    }
  },
  'google/gemma-2b-it': {
    name: 'Gemma 2B',
    skills: {
      "K.CC.C.7": "0.34",
      "K.NBT.A.1": "0.54",
      "K.OA.A.4": "0.78",
      "K.OA.A.5": "0.63",
      "1.OA.A.1": "0.75",
      "1.OA.A.2": "0.77",
      "1.OA.D.8": "0.53",
      "2.MD.B.5": "0.71",
      "2.MD.C.8": "0.72",
      "2.NBT.B.5": "0.71",
      "2.NBT.B.6": "0.39",
      "2.NBT.B.7": "0.67",
      "2.OA.A.1": "0.62",
      "3.MD.D.8-polygon": "0.27",
      "3.MD.D.8-quadrilateral": "0.41",
      "3.MD.D.8-triangle": "0.45",
      "3.NBT.A.2": "0.77",
      "3.OA.A.3": "0.74",
      "3.OA.A.4": "0.70",
      "3.OA.C.7": "0.71",
      "3.OA.D.8": "0.62",
      "4.MD.A.2-decimal": "0.37",
      "4.MD.A.2-fraction": "0.16",
      "4.MD.A.3": "0.77",
      "4.NBT.B.4": "0.66",
      "4.NBT.B.5": "0.30",
      "4.NBT.B.6": "0.15",
      "4.NF.A.2": "0.43",
      "4.OA.A.3": "0.08",
      "4.OA.B.4": "0.34",
      "5.NBT.B.5": "0.16",
      "5.NBT.B.6": "0.52",
      "5.NBT.B.7": "0.37",
      "5.NF.A.1": "0.02",
      "5.NF.A.2": "0.03",
      "5.NF.B.4": "0.26",
      "5.OA.A.1": "0.26",
      "6.EE.A.1": "0.82",
      "6.EE.B.7": "0.58",
      "6.NS.B.2": "0.24",
      "6.NS.B.3": "0.28",
      "7.NS.A.1-decimal": "0.56",
      "7.NS.A.1-fraction": "0.01",
      "7.NS.A.2": "0.09",
      "7.NS.A.3-decimal": "0.42",
      "7.NS.A.3-fraction": "0.02",
      "8.EE.A.2": "0.36",
      "8.EE.C.7": "0.09",
      "8.EE.C.8": "0.00",
    }
  },
  'google/gemma-7b-it': {
    name: 'Gemma 7B',
    skills: {
      "K.CC.C.7": "0.49",
      "K.NBT.A.1": "0.95",
      "K.OA.A.4": "0.86",
      "K.OA.A.5": "0.96",
      "1.OA.A.1": "0.97",
      "1.OA.A.2": "0.91",
      "1.OA.D.8": "0.88",
      "2.MD.B.5": "0.96",
      "2.MD.C.8": "0.95",
      "2.NBT.B.5": "0.94",
      "2.NBT.B.6": "0.85",
      "2.NBT.B.7": "0.85",
      "2.OA.A.1": "0.86",
      "3.MD.D.8-polygon": "0.63",
      "3.MD.D.8-quadrilateral": "0.58",
      "3.MD.D.8-triangle": "0.78",
      "3.NBT.A.2": "0.96",
      "3.OA.A.3": "0.95",
      "3.OA.A.4": "0.88",
      "3.OA.C.7": "0.91",
      "3.OA.D.8": "0.89",
      "4.MD.A.2-decimal": "0.66",
      "4.MD.A.2-fraction": "0.26",
      "4.MD.A.3": "0.50",
      "4.NBT.B.4": "0.94",
      "4.NBT.B.5": "0.55",
      "4.NBT.B.6": "0.40",
      "4.NF.A.2": "0.19",
      "4.OA.A.3": "0.43",
      "4.OA.B.4": "0.24",
      "5.NBT.B.5": "0.32",
      "5.NBT.B.6": "0.60",
      "5.NBT.B.7": "0.51",
      "5.NF.A.1": "0.00",
      "5.NF.A.2": "0.06",
      "5.NF.B.4": "0.28",
      "5.OA.A.1": "0.54",
      "6.EE.A.1": "0.92",
      "6.EE.B.7": "0.87",
      "6.NS.B.2": "0.33",
      "6.NS.B.3": "0.43",
      "7.NS.A.1-decimal": "0.91",
      "7.NS.A.1-fraction": "0.01",
      "7.NS.A.2": "0.07",
      "7.NS.A.3-decimal": "0.74",
      "7.NS.A.3-fraction": "0.03",
      "8.EE.A.2": "0.72",
      "8.EE.C.7": "0.11",
      "8.EE.C.8": "0.00",
    }
  },
  'meta-llama/Llama-3-8b-chat-hf': {
    name: 'Llama 3 8B',
    skills: {
      "K.CC.C.7": "0.46",
      "K.NBT.A.1": "0.92",
      "K.OA.A.4": "0.98",
      "K.OA.A.5": "0.93",
      "1.OA.A.1": "0.99",
      "1.OA.A.2": "0.96",
      "1.OA.D.8": "0.96",
      "2.MD.B.5": "0.98",
      "2.MD.C.8": "0.98",
      "2.NBT.B.5": "0.92",
      "2.NBT.B.6": "0.91",
      "2.NBT.B.7": "0.99",
      "2.OA.A.1": "0.97",
      "3.MD.D.8-polygon": "0.84",
      "3.MD.D.8-quadrilateral": "0.97",
      "3.MD.D.8-triangle": "0.99",
      "3.NBT.A.2": "0.98",
      "3.OA.A.3": "0.92",
      "3.OA.A.4": "0.97",
      "3.OA.C.7": "0.95",
      "3.OA.D.8": "0.92",
      "4.MD.A.2-decimal": "0.81",
      "4.MD.A.2-fraction": "0.35",
      "4.MD.A.3": "0.99",
      "4.NBT.B.4": "0.76",
      "4.NBT.B.5": "0.79",
      "4.NBT.B.6": "0.68",
      "4.NF.A.2": "0.48",
      "4.OA.A.3": "0.72",
      "4.OA.B.4": "0.97",
      "5.NBT.B.5": "0.43",
      "5.NBT.B.6": "0.93",
      "5.NBT.B.7": "0.58",
      "5.NF.A.1": "0.10",
      "5.NF.A.2": "0.19",
      "5.NF.B.4": "0.50",
      "5.OA.A.1": "0.83",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "0.99",
      "6.NS.B.2": "0.65",
      "6.NS.B.3": "0.51",
      "7.NS.A.1-decimal": "0.93",
      "7.NS.A.1-fraction": "0.08",
      "7.NS.A.2": "0.53",
      "7.NS.A.3-decimal": "0.77",
      "7.NS.A.3-fraction": "0.15",
      "8.EE.A.2": "0.87",
      "8.EE.C.7": "0.20",
      "8.EE.C.8": "0.00",
    }
  },
  'meta-llama/Llama-3-70b-chat-hf': {
    name: 'Llama 3 70B',
    skills: {
      "K.CC.C.7": "0.70",
      "K.NBT.A.1": "0.96",
      "K.OA.A.4": "0.95",
      "K.OA.A.5": "0.94",
      "1.OA.A.1": "0.97",
      "1.OA.A.2": "0.97",
      "1.OA.D.8": "0.98",
      "2.MD.B.5": "0.95",
      "2.MD.C.8": "0.99",
      "2.NBT.B.5": "0.98",
      "2.NBT.B.6": "0.95",
      "2.NBT.B.7": "0.99",
      "2.OA.A.1": "0.96",
      "3.MD.D.8-polygon": "0.92",
      "3.MD.D.8-quadrilateral": "1.00",
      "3.MD.D.8-triangle": "1.00",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.93",
      "3.OA.A.4": "1.00",
      "3.OA.C.7": "0.97",
      "3.OA.D.8": "0.94",
      "4.MD.A.2-decimal": "0.88",
      "4.MD.A.2-fraction": "0.43",
      "4.MD.A.3": "0.99",
      "4.NBT.B.4": "0.95",
      "4.NBT.B.5": "0.93",
      "4.NBT.B.6": "0.74",
      "4.NF.A.2": "0.54",
      "4.OA.A.3": "0.78",
      "4.OA.B.4": "0.94",
      "5.NBT.B.5": "0.71",
      "5.NBT.B.6": "1.00",
      "5.NBT.B.7": "0.69",
      "5.NF.A.1": "0.24",
      "5.NF.A.2": "0.57",
      "5.NF.B.4": "0.69",
      "5.OA.A.1": "0.94",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "0.99",
      "6.NS.B.2": "0.88",
      "6.NS.B.3": "0.60",
      "7.NS.A.1-decimal": "0.97",
      "7.NS.A.1-fraction": "0.40",
      "7.NS.A.2": "0.88",
      "7.NS.A.3-decimal": "0.85",
      "7.NS.A.3-fraction": "0.45",
      "8.EE.A.2": "0.99",
      "8.EE.C.7": "0.29",
      "8.EE.C.8": "0.00",
    }
  },
  'codellama/CodeLlama-7b-Instruct-hf': {
    name: 'CodeLlama 7B',
    skills: {
      "K.CC.C.7": "0.64",
      "K.NBT.A.1": "0.14",
      "K.OA.A.4": "0.21",
      "K.OA.A.5": "0.39",
      "1.OA.A.1": "0.21",
      "1.OA.A.2": "0.25",
      "1.OA.D.8": "0.24",
      "2.MD.B.5": "0.10",
      "2.MD.C.8": "0.40",
      "2.NBT.B.5": "0.21",
      "2.NBT.B.6": "0.21",
      "2.NBT.B.7": "0.22",
      "2.OA.A.1": "0.13",
      "3.MD.D.8-polygon": "0.00",
      "3.MD.D.8-quadrilateral": "0.01",
      "3.MD.D.8-triangle": "0.00",
      "3.NBT.A.2": "0.24",
      "3.OA.A.3": "0.18",
      "3.OA.A.4": "0.11",
      "3.OA.C.7": "0.13",
      "3.OA.D.8": "0.24",
      "4.MD.A.2-decimal": "0.15",
      "4.MD.A.2-fraction": "0.05",
      "4.MD.A.3": "0.03",
      "4.NBT.B.4": "0.12",
      "4.NBT.B.5": "0.05",
      "4.NBT.B.6": "0.03",
      "4.NF.A.2": "0.19",
      "4.OA.A.3": "0.05",
      "4.OA.B.4": "0.15",
      "5.NBT.B.5": "0.01",
      "5.NBT.B.6": "0.01",
      "5.NBT.B.7": "0.04",
      "5.NF.A.1": "0.00",
      "5.NF.A.2": "0.00",
      "5.NF.B.4": "0.11",
      "5.OA.A.1": "0.08",
      "6.EE.A.1": "0.10",
      "6.EE.B.7": "0.09",
      "6.NS.B.2": "0.00",
      "6.NS.B.3": "0.02",
      "7.NS.A.1-decimal": "0.03",
      "7.NS.A.1-fraction": "0.00",
      "7.NS.A.2": "0.04",
      "7.NS.A.3-decimal": "0.08",
      "7.NS.A.3-fraction": "0.00",
      "8.EE.A.2": "0.09",
      "8.EE.C.7": "0.01",
      "8.EE.C.8": "0.00",
    }
  },
  'codellama/CodeLlama-13b-Instruct-hf': {
    name: 'CodeLlama 13B',
    skills: {
      "K.CC.C.7": "0.53",
      "K.NBT.A.1": "0.84",
      "K.OA.A.4": "0.96",
      "K.OA.A.5": "0.89",
      "1.OA.A.1": "0.94",
      "1.OA.A.2": "0.96",
      "1.OA.D.8": "0.76",
      "2.MD.B.5": "0.87",
      "2.MD.C.8": "0.92",
      "2.NBT.B.5": "0.88",
      "2.NBT.B.6": "0.77",
      "2.NBT.B.7": "0.79",
      "2.OA.A.1": "0.77",
      "3.MD.D.8-polygon": "0.37",
      "3.MD.D.8-quadrilateral": "0.41",
      "3.MD.D.8-triangle": "0.47",
      "3.NBT.A.2": "0.89",
      "3.OA.A.3": "0.91",
      "3.OA.A.4": "0.80",
      "3.OA.C.7": "0.94",
      "3.OA.D.8": "0.81",
      "4.MD.A.2-decimal": "0.61",
      "4.MD.A.2-fraction": "0.25",
      "4.MD.A.3": "0.31",
      "4.NBT.B.4": "0.81",
      "4.NBT.B.5": "0.43",
      "4.NBT.B.6": "0.20",
      "4.NF.A.2": "0.15",
      "4.OA.A.3": "0.27",
      "4.OA.B.4": "0.19",
      "5.NBT.B.5": "0.20",
      "5.NBT.B.6": "0.42",
      "5.NBT.B.7": "0.47",
      "5.NF.A.1": "0.00",
      "5.NF.A.2": "0.02",
      "5.NF.B.4": "0.23",
      "5.OA.A.1": "0.40",
      "6.EE.A.1": "0.61",
      "6.EE.B.7": "0.72",
      "6.NS.B.2": "0.22",
      "6.NS.B.3": "0.38",
      "7.NS.A.1-decimal": "0.72",
      "7.NS.A.1-fraction": "0.02",
      "7.NS.A.2": "0.08",
      "7.NS.A.3-decimal": "0.55",
      "7.NS.A.3-fraction": "0.01",
      "8.EE.A.2": "0.58",
      "8.EE.C.7": "0.20",
      "8.EE.C.8": "0.00",
    }
  },
  'codellama/CodeLlama-34b-Instruct-hf': {
    name: 'CodeLlama 34B',
    skills: {
      "K.CC.C.7": "0.84",
      "K.NBT.A.1": "0.11",
      "K.OA.A.4": "0.21",
      "K.OA.A.5": "0.32",
      "1.OA.A.1": "0.16",
      "1.OA.A.2": "0.19",
      "1.OA.D.8": "0.10",
      "2.MD.B.5": "0.13",
      "2.MD.C.8": "0.34",
      "2.NBT.B.5": "0.27",
      "2.NBT.B.6": "0.27",
      "2.NBT.B.7": "0.24",
      "2.OA.A.1": "0.08",
      "3.MD.D.8-polygon": "0.01",
      "3.MD.D.8-quadrilateral": "0.04",
      "3.MD.D.8-triangle": "0.01",
      "3.NBT.A.2": "0.24",
      "3.OA.A.3": "0.35",
      "3.OA.A.4": "0.14",
      "3.OA.C.7": "0.33",
      "3.OA.D.8": "0.22",
      "4.MD.A.2-decimal": "0.20",
      "4.MD.A.2-fraction": "0.12",
      "4.MD.A.3": "0.27",
      "4.NBT.B.4": "0.19",
      "4.NBT.B.5": "0.25",
      "4.NBT.B.6": "0.09",
      "4.NF.A.2": "0.14",
      "4.OA.A.3": "0.08",
      "4.OA.B.4": "0.23",
      "5.NBT.B.5": "0.04",
      "5.NBT.B.6": "0.01",
      "5.NBT.B.7": "0.05",
      "5.NF.A.1": "0.00",
      "5.NF.A.2": "0.04",
      "5.NF.B.4": "0.15",
      "5.OA.A.1": "0.16",
      "6.EE.A.1": "0.24",
      "6.EE.B.7": "0.02",
      "6.NS.B.2": "0.00",
      "6.NS.B.3": "0.05",
      "7.NS.A.1-decimal": "0.33",
      "7.NS.A.1-fraction": "0.01",
      "7.NS.A.2": "0.06",
      "7.NS.A.3-decimal": "0.07",
      "7.NS.A.3-fraction": "0.00",
      "8.EE.A.2": "0.11",
      "8.EE.C.7": "0.07",
      "8.EE.C.8": "0.00",
    }
  },
  'microsoft/phi-2': {
    name: 'phi-2',
    skills: {
      "K.CC.C.7": "0.93",
      "K.NBT.A.1": "0.95",
      "K.OA.A.4": "0.99",
      "K.OA.A.5": "0.93",
      "1.OA.A.1": "0.98",
      "1.OA.A.2": "0.97",
      "1.OA.D.8": "0.93",
      "2.MD.B.5": "0.98",
      "2.MD.C.8": "0.97",
      "2.NBT.B.5": "0.98",
      "2.NBT.B.6": "0.78",
      "2.NBT.B.7": "0.67",
      "2.OA.A.1": "0.97",
      "3.MD.D.8-polygon": "0.56",
      "3.MD.D.8-quadrilateral": "0.68",
      "3.MD.D.8-triangle": "0.47",
      "3.NBT.A.2": "0.71",
      "3.OA.A.3": "0.96",
      "3.OA.A.4": "0.93",
      "3.OA.C.7": "0.97",
      "3.OA.D.8": "0.90",
      "4.MD.A.2-decimal": "0.72",
      "4.MD.A.2-fraction": "0.34",
      "4.MD.A.3": "0.73",
      "4.NBT.B.4": "0.22",
      "4.NBT.B.5": "0.63",
      "4.NBT.B.6": "0.25",
      "4.NF.A.2": "0.90",
      "4.OA.A.3": "0.37",
      "4.OA.B.4": "0.00",
      "5.NBT.B.5": "0.23",
      "5.NBT.B.6": "0.60",
      "5.NBT.B.7": "0.52",
      "5.NF.A.1": "0.04",
      "5.NF.A.2": "0.15",
      "5.NF.B.4": "0.53",
      "5.OA.A.1": "0.62",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "0.72",
      "6.NS.B.2": "0.25",
      "6.NS.B.3": "0.47",
      "7.NS.A.1-decimal": "0.54",
      "7.NS.A.1-fraction": "0.14",
      "7.NS.A.2": "0.55",
      "7.NS.A.3-decimal": "0.48",
      "7.NS.A.3-fraction": "0.11",
      "8.EE.A.2": "0.92",
      "8.EE.C.7": "0.29",
      "8.EE.C.8": "0.00",
    }
  },
  'mistralai/Mistral-7B-Instruct-v0.3': {
    name: 'Mistral 7B',
    skills: {
      "K.CC.C.7": "0.78",
      "K.NBT.A.1": "0.89",
      "K.OA.A.4": "0.90",
      "K.OA.A.5": "0.87",
      "1.OA.A.1": "0.91",
      "1.OA.A.2": "0.93",
      "1.OA.D.8": "0.93",
      "2.MD.B.5": "0.92",
      "2.MD.C.8": "0.93",
      "2.NBT.B.5": "0.95",
      "2.NBT.B.6": "0.79",
      "2.NBT.B.7": "0.89",
      "2.OA.A.1": "0.92",
      "3.MD.D.8-polygon": "0.56",
      "3.MD.D.8-quadrilateral": "0.83",
      "3.MD.D.8-triangle": "0.60",
      "3.NBT.A.2": "0.91",
      "3.OA.A.3": "0.88",
      "3.OA.A.4": "0.90",
      "3.OA.C.7": "0.90",
      "3.OA.D.8": "0.89",
      "4.MD.A.2-decimal": "0.70",
      "4.MD.A.2-fraction": "0.28",
      "4.MD.A.3": "0.51",
      "4.NBT.B.4": "0.89",
      "4.NBT.B.5": "0.77",
      "4.NBT.B.6": "0.46",
      "4.NF.A.2": "0.71",
      "4.OA.A.3": "0.54",
      "4.OA.B.4": "0.56",
      "5.NBT.B.5": "0.41",
      "5.NBT.B.6": "0.58",
      "5.NBT.B.7": "0.55",
      "5.NF.A.1": "0.06",
      "5.NF.A.2": "0.14",
      "5.NF.B.4": "0.52",
      "5.OA.A.1": "0.59",
      "6.EE.A.1": "0.89",
      "6.EE.B.7": "0.81",
      "6.NS.B.2": "0.37",
      "6.NS.B.3": "0.42",
      "7.NS.A.1-decimal": "0.91",
      "7.NS.A.1-fraction": "0.06",
      "7.NS.A.2": "0.33",
      "7.NS.A.3-decimal": "0.77",
      "7.NS.A.3-fraction": "0.08",
      "8.EE.A.2": "0.83",
      "8.EE.C.7": "0.26",
      "8.EE.C.8": "0.00",
    }
  },
  'mistralai/Mixtral-8x7B-Instruct-v0.1': {
    name: 'Mixtral 8x7B',
    skills: {
      "K.CC.C.7": "0.95",
      "K.NBT.A.1": "0.88",
      "K.OA.A.4": "0.92",
      "K.OA.A.5": "0.87",
      "1.OA.A.1": "0.95",
      "1.OA.A.2": "0.93",
      "1.OA.D.8": "0.96",
      "2.MD.B.5": "0.95",
      "2.MD.C.8": "0.94",
      "2.NBT.B.5": "0.88",
      "2.NBT.B.6": "0.83",
      "2.NBT.B.7": "0.89",
      "2.OA.A.1": "0.92",
      "3.MD.D.8-polygon": "0.73",
      "3.MD.D.8-quadrilateral": "0.89",
      "3.MD.D.8-triangle": "0.92",
      "3.NBT.A.2": "0.88",
      "3.OA.A.3": "0.94",
      "3.OA.A.4": "0.90",
      "3.OA.C.7": "0.94",
      "3.OA.D.8": "0.90",
      "4.MD.A.2-decimal": "0.83",
      "4.MD.A.2-fraction": "0.32",
      "4.MD.A.3": "0.99",
      "4.NBT.B.4": "0.89",
      "4.NBT.B.5": "0.90",
      "4.NBT.B.6": "0.49",
      "4.NF.A.2": "0.73",
      "4.OA.A.3": "0.45",
      "4.OA.B.4": "0.86",
      "5.NBT.B.5": "0.54",
      "5.NBT.B.6": "0.92",
      "5.NBT.B.7": "0.57",
      "5.NF.A.1": "0.09",
      "5.NF.A.2": "0.10",
      "5.NF.B.4": "0.49",
      "5.OA.A.1": "0.68",
      "6.EE.A.1": "0.91",
      "6.EE.B.7": "0.90",
      "6.NS.B.2": "0.74",
      "6.NS.B.3": "0.53",
      "7.NS.A.1-decimal": "0.92",
      "7.NS.A.1-fraction": "0.08",
      "7.NS.A.2": "0.48",
      "7.NS.A.3-decimal": "0.78",
      "7.NS.A.3-fraction": "0.12",
      "8.EE.A.2": "0.95",
      "8.EE.C.7": "0.19",
      "8.EE.C.8": "0.00",
    }
  },
  'mistralai/Mixtral-8x22B-Instruct-v0.1': {
    name: 'Mixtral 8x22B',
    skills: {
      "K.CC.C.7": "0.53",
      "K.NBT.A.1": "0.94",
      "K.OA.A.4": "0.97",
      "K.OA.A.5": "0.95",
      "1.OA.A.1": "0.98",
      "1.OA.A.2": "0.98",
      "1.OA.D.8": "1.00",
      "2.MD.B.5": "0.97",
      "2.MD.C.8": "0.99",
      "2.NBT.B.5": "0.97",
      "2.NBT.B.6": "0.96",
      "2.NBT.B.7": "0.99",
      "2.OA.A.1": "0.98",
      "3.MD.D.8-polygon": "0.85",
      "3.MD.D.8-quadrilateral": "0.98",
      "3.MD.D.8-triangle": "0.99",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.94",
      "3.OA.A.4": "1.00",
      "3.OA.C.7": "0.97",
      "3.OA.D.8": "0.96",
      "4.MD.A.2-decimal": "0.88",
      "4.MD.A.2-fraction": "0.34",
      "4.MD.A.3": "0.98",
      "4.NBT.B.4": "0.97",
      "4.NBT.B.5": "0.95",
      "4.NBT.B.6": "0.74",
      "4.NF.A.2": "0.58",
      "4.OA.A.3": "0.72",
      "4.OA.B.4": "0.43",
      "5.NBT.B.5": "0.76",
      "5.NBT.B.6": "0.99",
      "5.NBT.B.7": "0.67",
      "5.NF.A.1": "0.28",
      "5.NF.A.2": "0.29",
      "5.NF.B.4": "0.62",
      "5.OA.A.1": "0.90",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "0.94",
      "6.NS.B.2": "0.94",
      "6.NS.B.3": "0.64",
      "7.NS.A.1-decimal": "0.98",
      "7.NS.A.1-fraction": "0.48",
      "7.NS.A.2": "0.74",
      "7.NS.A.3-decimal": "0.87",
      "7.NS.A.3-fraction": "0.22",
      "8.EE.A.2": "0.97",
      "8.EE.C.7": "0.34",
      "8.EE.C.8": "0.00",
    }
  },
  'gpt-4o-2024-05-13': {
    name: 'GPT-4o',
    skills: {
      "K.CC.C.7": "1.00",
      "K.NBT.A.1": "0.98",
      "K.OA.A.4": "0.98",
      "K.OA.A.5": "0.97",
      "1.OA.A.1": "0.98",
      "1.OA.A.2": "0.97",
      "1.OA.D.8": "1.00",
      "2.MD.B.5": "0.98",
      "2.MD.C.8": "0.98",
      "2.NBT.B.5": "0.98",
      "2.NBT.B.6": "0.96",
      "2.NBT.B.7": "1.00",
      "2.OA.A.1": "0.98",
      "3.MD.D.8-polygon": "0.96",
      "3.MD.D.8-quadrilateral": "1.00",
      "3.MD.D.8-triangle": "1.00",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.95",
      "3.OA.A.4": "0.98",
      "3.OA.C.7": "0.98",
      "3.OA.D.8": "0.96",
      "4.MD.A.2-decimal": "0.95",
      "4.MD.A.2-fraction": "0.54",
      "4.MD.A.3": "1.00",
      "4.NBT.B.4": "1.00",
      "4.NBT.B.5": "0.99",
      "4.NBT.B.6": "0.97",
      "4.NF.A.2": "0.99",
      "4.OA.A.3": "0.80",
      "4.OA.B.4": "1.00",
      "5.NBT.B.5": "0.97",
      "5.NBT.B.6": "1.00",
      "5.NBT.B.7": "0.80",
      "5.NF.A.1": "0.75",
      "5.NF.A.2": "0.81",
      "5.NF.B.4": "0.76",
      "5.OA.A.1": "0.94",
      "6.EE.A.1": "1.00",
      "6.EE.B.7": "0.99",
      "6.NS.B.2": "0.96",
      "6.NS.B.3": "0.82",
      "7.NS.A.1-decimal": "0.99",
      "7.NS.A.1-fraction": "0.82",
      "7.NS.A.2": "0.92",
      "7.NS.A.3-decimal": "0.91",
      "7.NS.A.3-fraction": "0.71",
      "8.EE.A.2": "0.98",
      "8.EE.C.7": "0.78",
      "8.EE.C.8": "0.00",
    }
  },
  'gpt-3.5-turbo-0125': {
    name: 'GPT-3.5 Turbo',
    skills: {
      "K.CC.C.7": "0.99",
      "K.NBT.A.1": "0.94",
      "K.OA.A.4": "0.97",
      "K.OA.A.5": "0.93",
      "1.OA.A.1": "0.99",
      "1.OA.A.2": "0.96",
      "1.OA.D.8": "0.99",
      "2.MD.B.5": "0.99",
      "2.MD.C.8": "0.98",
      "2.NBT.B.5": "0.97",
      "2.NBT.B.6": "0.98",
      "2.NBT.B.7": "0.98",
      "2.OA.A.1": "0.96",
      "3.MD.D.8-polygon": "0.95",
      "3.MD.D.8-quadrilateral": "0.99",
      "3.MD.D.8-triangle": "0.99",
      "3.NBT.A.2": "1.00",
      "3.OA.A.3": "0.95",
      "3.OA.A.4": "0.97",
      "3.OA.C.7": "0.98",
      "3.OA.D.8": "0.93",
      "4.MD.A.2-decimal": "0.92",
      "4.MD.A.2-fraction": "0.45",
      "4.MD.A.3": "1.00",
      "4.NBT.B.4": "1.00",
      "4.NBT.B.5": "0.95",
      "4.NBT.B.6": "0.79",
      "4.NF.A.2": "0.89",
      "4.OA.A.3": "0.68",
      "4.OA.B.4": "0.98",
      "5.NBT.B.5": "0.81",
      "5.NBT.B.6": "1.00",
      "5.NBT.B.7": "0.69",
      "5.NF.A.1": "0.42",
      "5.NF.A.2": "0.73",
      "5.NF.B.4": "0.73",
      "5.OA.A.1": "0.96",
      "6.EE.A.1": "0.89",
      "6.EE.B.7": "0.99",
      "6.NS.B.2": "0.96",
      "6.NS.B.3": "0.70",
      "7.NS.A.1-decimal": "0.99",
      "7.NS.A.1-fraction": "0.62",
      "7.NS.A.2": "0.84",
      "7.NS.A.3-decimal": "0.91",
      "7.NS.A.3-fraction": "0.33",
      "8.EE.A.2": "0.99",
      "8.EE.C.7": "0.46",
      "8.EE.C.8": "0.03",
    }
  },


}

function updateTable() {
  const model1 = document.getElementById('model1').value;
  const model2 = document.getElementById('model2').value;

  const tableBody = document.getElementById('comparisonTable').getElementsByTagName('tbody')[0];
  const model1Header = document.getElementById('model1Header');
  const model2Header = document.getElementById('model2Header');

  tableBody.innerHTML = '';
  model1Header.textContent = model1 ? modelData[model1]?.name || '' : '';
  model2Header.textContent = model2 ? modelData[model2]?.name || '' : '';

  const skills = new Set([
    ...(modelData[model1]?.skills ? Object.keys(modelData[model1].skills) : []),
    ...(modelData[model2]?.skills ? Object.keys(modelData[model2].skills) : [])
  ]);

  skills.forEach(skill => {
    const row = tableBody.insertRow();
    const skillCell = row.insertCell(0);
    const descCell = row.insertCell(1)
    const model1Cell = row.insertCell(2);
    const model2Cell = row.insertCell(3);


    skillCell.textContent = skill;
    descCell.textContent = standardData[skill];
    model1Cell.textContent = model1 ? modelData[model1]?.skills[skill] || '' : '';
    model2Cell.textContent = model2 ? modelData[model2]?.skills[skill] || '' : '';
  });
}
