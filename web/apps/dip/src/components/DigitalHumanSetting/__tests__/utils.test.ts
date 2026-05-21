import { describe, expect, it } from 'vitest'

import { DESettingMenuKey } from '../types'
import { deSettingMenuKeyOrder, getDeSettingMenuItems } from '../utils'

describe('DigitalHumanSetting/utils', () => {
  it('菜单项顺序与 key 正确', () => {
    expect(deSettingMenuKeyOrder).toEqual([
      DESettingMenuKey.BASIC,
      DESettingMenuKey.SKILL,
      DESettingMenuKey.KNOWLEDGE,
      DESettingMenuKey.CHANNEL,
    ])
  })

  it('菜单文案使用 i18n key（测试环境下 intl mock 返回 key）', () => {
    expect(getDeSettingMenuItems().map((item) => item.label)).toEqual([
      'digitalHuman.setting.menuBasic',
      'digitalHuman.setting.menuSkill',
      'digitalHuman.setting.menuKnowledge',
      'digitalHuman.setting.menuChannel',
    ])
  })

  it('getDeSettingMenuItems 在默认环境下与 deSettingMenuKeyOrder 长度一致', () => {
    expect(getDeSettingMenuItems().map((item) => item.key)).toEqual(deSettingMenuKeyOrder)
  })

  it('创建态可隐藏知识配置菜单', () => {
    expect(getDeSettingMenuItems({ hideKnowledge: true }).map((item) => item.key)).toEqual([
      DESettingMenuKey.BASIC,
      DESettingMenuKey.SKILL,
      DESettingMenuKey.CHANNEL,
    ])
  })

  it('查看态隐藏知识配置菜单时顺序保持稳定', () => {
    expect(getDeSettingMenuItems({ hideKnowledge: true }).map((item) => item.label)).toEqual([
      'digitalHuman.setting.menuBasic',
      'digitalHuman.setting.menuSkill',
      'digitalHuman.setting.menuChannel',
    ])
  })
})
