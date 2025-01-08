from pydantic import BaseModel

class SPMConfig(BaseModel):
    size: int
    delay: int

class TPUConfig(BaseModel):
    flops: int

class LSUConfig(BaseModel):
    width: int

class CoreConfig(BaseModel):
    type: str
    x: int
    y: int
    width: int
    spm: SPMConfig
    tpu: TPUConfig
    lsu: LSUConfig

class RouterConfig(BaseModel):
    type: str
    vc: int

class LinkConfig(BaseModel):
    width: int
    delay: int

class NoCConfig(BaseModel):
    type: str
    x: int
    y: int
    router: RouterConfig
    link: LinkConfig

class MemConfig(BaseModel):
    width: int
    delay: int

class ArchConfig(BaseModel):
    core: CoreConfig
    noc: NoCConfig
    mem: MemConfig

class ScratchpadConfig(BaseModel):
    size: int
    delay: int

