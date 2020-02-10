# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gym_env.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gym_env.proto',
  package='deepx',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\rgym_env.proto\x12\x05\x64\x65\x65px\"\x15\n\x08\x44iscrete\x12\t\n\x01n\x18\x01 \x01(\x05\"/\n\x03\x42ox\x12\r\n\x05shape\x18\x01 \x03(\x05\x12\x0b\n\x03low\x18\x02 \x03(\x02\x12\x0c\n\x04high\x18\x03 \x03(\x02\"C\n\x05Space\x12!\n\x08\x64iscrete\x18\x01 \x01(\x0b\x32\x0f.deepx.Discrete\x12\x17\n\x03\x62ox\x18\x02 \x01(\x0b\x32\n.deepx.Box\"\x14\n\x05State\x12\x0b\n\x03obs\x18\x01 \x03(\x02\"A\n\x04Step\x12\x1b\n\x05state\x18\x01 \x01(\x0b\x32\x0c.deepx.State\x12\x0e\n\x06reward\x18\x02 \x01(\x02\x12\x0c\n\x04\x64one\x18\x03 \x01(\x08\"z\n\tEnvConfig\x12\x0e\n\x06\x65nv_id\x18\x01 \x01(\t\x12\x10\n\x08num_envs\x18\x02 \x01(\x05\x12\"\n\x0c\x61\x63tion_space\x18\x03 \x01(\x0b\x32\x0c.deepx.Space\x12\'\n\x11observation_space\x18\x04 \x01(\x0b\x32\x0c.deepx.Space\"-\n\rCreateRequest\x12\n\n\x02id\x18\x01 \x01(\t\x12\x10\n\x08num_envs\x18\x02 \x01(\x05\"2\n\x0e\x43reateResponse\x12 \n\x06\x63onfig\x18\x01 \x01(\x0b\x32\x10.deepx.EnvConfig\"\x1e\n\x0cResetRequest\x12\x0e\n\x06\x65nv_id\x18\x01 \x01(\t\"-\n\rResetResponse\x12\x1c\n\x06states\x18\x01 \x03(\x0b\x32\x0c.deepx.State\"-\n\x0bStepRequest\x12\x0e\n\x06\x65nv_id\x18\x01 \x01(\t\x12\x0e\n\x06\x61\x63tion\x18\x02 \x03(\x05\"*\n\x0cStepResponse\x12\x1a\n\x05steps\x18\x01 \x03(\x0b\x32\x0b.deepx.Step\"\x1e\n\x0c\x43loseRequest\x12\x0e\n\x06\x65nv_id\x18\x01 \x01(\t\"\r\n\x0bListRequest\"\x1f\n\x0cListResponse\x12\x0f\n\x07\x65nv_ids\x18\x01 \x03(\t\"\x0f\n\rCloseResponse2\x86\x02\n\x03Gym\x12\x35\n\x06\x43reate\x12\x14.deepx.CreateRequest\x1a\x15.deepx.CreateResponse\x12\x32\n\x05Reset\x12\x13.deepx.ResetRequest\x1a\x14.deepx.ResetResponse\x12/\n\x04Step\x12\x12.deepx.StepRequest\x1a\x13.deepx.StepResponse\x12\x32\n\x05\x43lose\x12\x13.deepx.CloseRequest\x1a\x14.deepx.CloseResponse\x12/\n\x04List\x12\x12.deepx.ListRequest\x1a\x13.deepx.ListResponseb\x06proto3')
)




_DISCRETE = _descriptor.Descriptor(
  name='Discrete',
  full_name='deepx.Discrete',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='n', full_name='deepx.Discrete.n', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=24,
  serialized_end=45,
)


_BOX = _descriptor.Descriptor(
  name='Box',
  full_name='deepx.Box',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='shape', full_name='deepx.Box.shape', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='low', full_name='deepx.Box.low', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='high', full_name='deepx.Box.high', index=2,
      number=3, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=47,
  serialized_end=94,
)


_SPACE = _descriptor.Descriptor(
  name='Space',
  full_name='deepx.Space',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='discrete', full_name='deepx.Space.discrete', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='box', full_name='deepx.Space.box', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=96,
  serialized_end=163,
)


_STATE = _descriptor.Descriptor(
  name='State',
  full_name='deepx.State',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='obs', full_name='deepx.State.obs', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=165,
  serialized_end=185,
)


_STEP = _descriptor.Descriptor(
  name='Step',
  full_name='deepx.Step',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='state', full_name='deepx.Step.state', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='reward', full_name='deepx.Step.reward', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='done', full_name='deepx.Step.done', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=187,
  serialized_end=252,
)


_ENVCONFIG = _descriptor.Descriptor(
  name='EnvConfig',
  full_name='deepx.EnvConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_id', full_name='deepx.EnvConfig.env_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_envs', full_name='deepx.EnvConfig.num_envs', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action_space', full_name='deepx.EnvConfig.action_space', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='observation_space', full_name='deepx.EnvConfig.observation_space', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=254,
  serialized_end=376,
)


_CREATEREQUEST = _descriptor.Descriptor(
  name='CreateRequest',
  full_name='deepx.CreateRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='deepx.CreateRequest.id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_envs', full_name='deepx.CreateRequest.num_envs', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=378,
  serialized_end=423,
)


_CREATERESPONSE = _descriptor.Descriptor(
  name='CreateResponse',
  full_name='deepx.CreateResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='config', full_name='deepx.CreateResponse.config', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=425,
  serialized_end=475,
)


_RESETREQUEST = _descriptor.Descriptor(
  name='ResetRequest',
  full_name='deepx.ResetRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_id', full_name='deepx.ResetRequest.env_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=477,
  serialized_end=507,
)


_RESETRESPONSE = _descriptor.Descriptor(
  name='ResetResponse',
  full_name='deepx.ResetResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='states', full_name='deepx.ResetResponse.states', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=509,
  serialized_end=554,
)


_STEPREQUEST = _descriptor.Descriptor(
  name='StepRequest',
  full_name='deepx.StepRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_id', full_name='deepx.StepRequest.env_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='action', full_name='deepx.StepRequest.action', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=556,
  serialized_end=601,
)


_STEPRESPONSE = _descriptor.Descriptor(
  name='StepResponse',
  full_name='deepx.StepResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='steps', full_name='deepx.StepResponse.steps', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=603,
  serialized_end=645,
)


_CLOSEREQUEST = _descriptor.Descriptor(
  name='CloseRequest',
  full_name='deepx.CloseRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_id', full_name='deepx.CloseRequest.env_id', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=647,
  serialized_end=677,
)


_LISTREQUEST = _descriptor.Descriptor(
  name='ListRequest',
  full_name='deepx.ListRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=679,
  serialized_end=692,
)


_LISTRESPONSE = _descriptor.Descriptor(
  name='ListResponse',
  full_name='deepx.ListResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='env_ids', full_name='deepx.ListResponse.env_ids', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=694,
  serialized_end=725,
)


_CLOSERESPONSE = _descriptor.Descriptor(
  name='CloseResponse',
  full_name='deepx.CloseResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=727,
  serialized_end=742,
)

_SPACE.fields_by_name['discrete'].message_type = _DISCRETE
_SPACE.fields_by_name['box'].message_type = _BOX
_STEP.fields_by_name['state'].message_type = _STATE
_ENVCONFIG.fields_by_name['action_space'].message_type = _SPACE
_ENVCONFIG.fields_by_name['observation_space'].message_type = _SPACE
_CREATERESPONSE.fields_by_name['config'].message_type = _ENVCONFIG
_RESETRESPONSE.fields_by_name['states'].message_type = _STATE
_STEPRESPONSE.fields_by_name['steps'].message_type = _STEP
DESCRIPTOR.message_types_by_name['Discrete'] = _DISCRETE
DESCRIPTOR.message_types_by_name['Box'] = _BOX
DESCRIPTOR.message_types_by_name['Space'] = _SPACE
DESCRIPTOR.message_types_by_name['State'] = _STATE
DESCRIPTOR.message_types_by_name['Step'] = _STEP
DESCRIPTOR.message_types_by_name['EnvConfig'] = _ENVCONFIG
DESCRIPTOR.message_types_by_name['CreateRequest'] = _CREATEREQUEST
DESCRIPTOR.message_types_by_name['CreateResponse'] = _CREATERESPONSE
DESCRIPTOR.message_types_by_name['ResetRequest'] = _RESETREQUEST
DESCRIPTOR.message_types_by_name['ResetResponse'] = _RESETRESPONSE
DESCRIPTOR.message_types_by_name['StepRequest'] = _STEPREQUEST
DESCRIPTOR.message_types_by_name['StepResponse'] = _STEPRESPONSE
DESCRIPTOR.message_types_by_name['CloseRequest'] = _CLOSEREQUEST
DESCRIPTOR.message_types_by_name['ListRequest'] = _LISTREQUEST
DESCRIPTOR.message_types_by_name['ListResponse'] = _LISTRESPONSE
DESCRIPTOR.message_types_by_name['CloseResponse'] = _CLOSERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Discrete = _reflection.GeneratedProtocolMessageType('Discrete', (_message.Message,), {
  'DESCRIPTOR' : _DISCRETE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.Discrete)
  })
_sym_db.RegisterMessage(Discrete)

Box = _reflection.GeneratedProtocolMessageType('Box', (_message.Message,), {
  'DESCRIPTOR' : _BOX,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.Box)
  })
_sym_db.RegisterMessage(Box)

Space = _reflection.GeneratedProtocolMessageType('Space', (_message.Message,), {
  'DESCRIPTOR' : _SPACE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.Space)
  })
_sym_db.RegisterMessage(Space)

State = _reflection.GeneratedProtocolMessageType('State', (_message.Message,), {
  'DESCRIPTOR' : _STATE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.State)
  })
_sym_db.RegisterMessage(State)

Step = _reflection.GeneratedProtocolMessageType('Step', (_message.Message,), {
  'DESCRIPTOR' : _STEP,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.Step)
  })
_sym_db.RegisterMessage(Step)

EnvConfig = _reflection.GeneratedProtocolMessageType('EnvConfig', (_message.Message,), {
  'DESCRIPTOR' : _ENVCONFIG,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.EnvConfig)
  })
_sym_db.RegisterMessage(EnvConfig)

CreateRequest = _reflection.GeneratedProtocolMessageType('CreateRequest', (_message.Message,), {
  'DESCRIPTOR' : _CREATEREQUEST,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.CreateRequest)
  })
_sym_db.RegisterMessage(CreateRequest)

CreateResponse = _reflection.GeneratedProtocolMessageType('CreateResponse', (_message.Message,), {
  'DESCRIPTOR' : _CREATERESPONSE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.CreateResponse)
  })
_sym_db.RegisterMessage(CreateResponse)

ResetRequest = _reflection.GeneratedProtocolMessageType('ResetRequest', (_message.Message,), {
  'DESCRIPTOR' : _RESETREQUEST,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.ResetRequest)
  })
_sym_db.RegisterMessage(ResetRequest)

ResetResponse = _reflection.GeneratedProtocolMessageType('ResetResponse', (_message.Message,), {
  'DESCRIPTOR' : _RESETRESPONSE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.ResetResponse)
  })
_sym_db.RegisterMessage(ResetResponse)

StepRequest = _reflection.GeneratedProtocolMessageType('StepRequest', (_message.Message,), {
  'DESCRIPTOR' : _STEPREQUEST,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.StepRequest)
  })
_sym_db.RegisterMessage(StepRequest)

StepResponse = _reflection.GeneratedProtocolMessageType('StepResponse', (_message.Message,), {
  'DESCRIPTOR' : _STEPRESPONSE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.StepResponse)
  })
_sym_db.RegisterMessage(StepResponse)

CloseRequest = _reflection.GeneratedProtocolMessageType('CloseRequest', (_message.Message,), {
  'DESCRIPTOR' : _CLOSEREQUEST,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.CloseRequest)
  })
_sym_db.RegisterMessage(CloseRequest)

ListRequest = _reflection.GeneratedProtocolMessageType('ListRequest', (_message.Message,), {
  'DESCRIPTOR' : _LISTREQUEST,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.ListRequest)
  })
_sym_db.RegisterMessage(ListRequest)

ListResponse = _reflection.GeneratedProtocolMessageType('ListResponse', (_message.Message,), {
  'DESCRIPTOR' : _LISTRESPONSE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.ListResponse)
  })
_sym_db.RegisterMessage(ListResponse)

CloseResponse = _reflection.GeneratedProtocolMessageType('CloseResponse', (_message.Message,), {
  'DESCRIPTOR' : _CLOSERESPONSE,
  '__module__' : 'gym_env_pb2'
  # @@protoc_insertion_point(class_scope:deepx.CloseResponse)
  })
_sym_db.RegisterMessage(CloseResponse)



_GYM = _descriptor.ServiceDescriptor(
  name='Gym',
  full_name='deepx.Gym',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=745,
  serialized_end=1007,
  methods=[
  _descriptor.MethodDescriptor(
    name='Create',
    full_name='deepx.Gym.Create',
    index=0,
    containing_service=None,
    input_type=_CREATEREQUEST,
    output_type=_CREATERESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Reset',
    full_name='deepx.Gym.Reset',
    index=1,
    containing_service=None,
    input_type=_RESETREQUEST,
    output_type=_RESETRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Step',
    full_name='deepx.Gym.Step',
    index=2,
    containing_service=None,
    input_type=_STEPREQUEST,
    output_type=_STEPRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='Close',
    full_name='deepx.Gym.Close',
    index=3,
    containing_service=None,
    input_type=_CLOSEREQUEST,
    output_type=_CLOSERESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='List',
    full_name='deepx.Gym.List',
    index=4,
    containing_service=None,
    input_type=_LISTREQUEST,
    output_type=_LISTRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GYM)

DESCRIPTOR.services_by_name['Gym'] = _GYM

# @@protoc_insertion_point(module_scope)
