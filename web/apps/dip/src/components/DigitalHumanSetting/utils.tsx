import intl from 'react-intl-universal'
import { type DESettingMenuItem, DESettingMenuKey } from './types'

/** 侧栏菜单 key 顺序（与原先 `deSettingMenuItems` 一致） */
export const deSettingMenuKeyOrder: DESettingMenuKey[] = [
  DESettingMenuKey.BASIC,
  DESettingMenuKey.SKILL,
  DESettingMenuKey.KNOWLEDGE,
  DESettingMenuKey.CHANNEL,
]

/** 各菜单项对应的 i18n 文案 key */
const labelKeyByMenuKey: Record<DESettingMenuKey, string> = {
  [DESettingMenuKey.BASIC]: 'digitalHuman.setting.menuBasic',
  [DESettingMenuKey.SKILL]: 'digitalHuman.setting.menuSkill',
  [DESettingMenuKey.KNOWLEDGE]: 'digitalHuman.setting.menuKnowledge',
  [DESettingMenuKey.CHANNEL]: 'digitalHuman.setting.menuChannel',
}

interface GetDeSettingMenuItemsOptions {
  hideKnowledge?: boolean
}

/** 侧栏菜单项 */
export const getDeSettingMenuItems = (
  options: GetDeSettingMenuItemsOptions = {},
): DESettingMenuItem[] => {
  const { hideKnowledge = false } = options

  return deSettingMenuKeyOrder
    .filter((key) => !hideKnowledge || key !== DESettingMenuKey.KNOWLEDGE)
    .map((key) => ({
    key,
    label: intl.get(labelKeyByMenuKey[key]),
    }))
}
