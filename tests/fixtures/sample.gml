graph [
  directed 1
  node [
    id 0
    label "api"
    graphics [
      fill "blue"
    ]
    LabelGraphics [
      text "API Gateway"
    ]
    type "service"
  ]
  node [
    id 1
    label "auth"
    graphics [
      fill "red"
    ]
    LabelGraphics [
      text "Auth Service"
    ]
    type "service"
  ]
  node [
    id 2
    label "cache"
    graphics [
      fill "green"
    ]
    LabelGraphics [
      text "Cache Layer"
    ]
    type "infrastructure"
  ]
  node [
    id 3
    label "database"
    graphics [
      fill "orange"
    ]
    LabelGraphics [
      text "Database"
    ]
    type "infrastructure"
  ]
  node [
    id 4
    label "logger"
    graphics [
      fill "gray"
    ]
    LabelGraphics [
      text "Logger"
    ]
    type "utility"
  ]
  node [
    id 5
    label "config"
    graphics [
      fill "purple"
    ]
    LabelGraphics [
      text "Config Manager"
    ]
    type "utility"
  ]
  edge [
    source 0
    target 1
    label "authenticates"
    weight 1.0
  ]
  edge [
    source 0
    target 2
    label "reads"
    weight 0.8
  ]
  edge [
    source 0
    target 4
    weight 0.5
  ]
  edge [
    source 1
    target 3
    label "queries"
    weight 1.0
  ]
  edge [
    source 1
    target 4
    weight 0.3
  ]
  edge [
    source 2
    target 3
    label "syncs"
    weight 0.9
  ]
  edge [
    source 5
    target 0
    label "configures"
    weight 0.2
  ]
]
