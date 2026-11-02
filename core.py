roles = {
    "null": 0,
    "you": 1,
    "self": 2,
    "system": 3
}# 每一个角色

# 角色类
class Role:
    def __init__(self, role, roledict=roles):
        # 对于任意类型的role采取不同转换
        match role:
            case str(): self.role = roledict.get(role, 0)
            case int(): self.role = role # 以便扩展，无需检测是否符合特点条件
            case _:
                raise TypeError(f"{type(role)} cannot be converted into a Role")
    def __eq__(self, other):
        return isinstance(other, Role) and self.role == other.role
    def __hash__(self):
        return hash(self.role)
# 单条对话类
class Dialogue:
    def __init__(self, role: Role, text: str):
        self.role = role
        self.text = text
        self.enable = True
    # 人称视角切换
    def person(self, mapping: dict):
        o = Dialogue(Role(0), self.text)
        # 根据提供的映射字典进行角色切换
        new_role = mapping.get(self.role, self.role)
        # 如果映射结果是 Role 实例，直接使用；否则构造为 Role
        if isinstance(new_role, Role): o.role = new_role
        else: o.role = Role(new_role)
        o.enable = self.enable
        return o
    # 变成第二人称视角
    def second_person(self):
        return self.person({
            Role(1): Role(2),
            Role(2): Role(1),
        })
# 将Dialogue列表转为标准OpenAI格式列表
def to_messages(dialogues, *, role_map=None, handle=lambda d, role, content: None):
    result = []
    if role_map is None:
        role_map = {1: "user", 2: "assistant"}
    for d in dialogues:
        if not d.enable:
            continue
        role = role_map.get(d.role.role, "system")
        content = d.text
        handle(d, role, content)
        result.append({
            "role": role,
            "content": content
        })
    return result

# 压缩提示词
compressing_prompt_words = Dialogue(Role("you"), "Compress our conversation and only output the compressed results, without carrying any irrelevant information")

class Participant:
    def __init__(self, name: str, context_length: int=524288):
        self.token_num = 0
        self.name = name
        self.context_length = context_length
        self.history = []
    def send(self, d: Dialogue, *, check=True):
        if check: self.check_compression()
        self.history.append(d)
    def ask(self) -> Dialogue:
        raise NotImplementedError("Subclasses must implement \"ask\"")
    # 压缩上下文(如果需要切换压缩方式可继承修改)
    def compression_context(self) -> Dialogue:
        self.history.append(compressing_prompt_words)
        try:
            c = self.ask()
        finally:
            self.history.pop()
        c.role = Role("system")
        for i in self.history:
            i.enable = False
        return c
    # 检测是否需要压缩(以后不需要压缩可以直接继承然后pass)
    # 默认按字数压缩，后面若实现ask后可通过token_num进行真实的计数
    # 占位实现: 建议继承后实现它
    def check_compression(self):
        if sum(len(h.text) for h in self.history if h.enable) >= self.context_length:
            self.send(self.compression_context(), check=False)

# 以下全部为测试 非全部功能
if __name__ == "__main__":
    import os
    from openai import OpenAI

    class OpenAIParticipant(Participant):
        def __init__(self, name: str, context_length: int=524288):
            super().__init__(name, context_length)
            self.model = os.environ.get('AI_MODEL')
            self.client = OpenAI(api_key=os.environ.get('AI_API_KEY'), base_url=os.environ.get('AI_BASE_URL'))
        def ask(self) -> Dialogue:
            return Dialogue(Role("self"), self.client.chat.completions.create(
                model=self.model,
                messages=to_messages(self.history),
            ).choices[0].message.content)

    a = OpenAIParticipant("a")
    b = OpenAIParticipant("b")
    a.send(Dialogue(Role(3), "辩题为Deepseek是否是蓝色大肥鱼，每次发言应简短保持在几句以内，你是正方"))
    b.send(Dialogue(Role(3), "辩题为Deepseek是否是蓝色大肥鱼，每次发言应简短保持在几句以内，你是反方"))
    print("======辩论开始(按Ctrl+C退出)======")
    try:
        for _ in range(20):
            a_d = a.ask()
            print(f"正方: \033[31m{a_d.text}\033[0m")
            a.send(a_d)
            b.send(a_d.second_person())
            b_d = b.ask()
            b.send(b_d)
            a.send(b_d.second_person())
            print(f"反方: \033[34m{b_d.text}\033[0m")
    except KeyboardInterrupt:
        print("quit")
